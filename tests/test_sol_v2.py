"""The Sol v2 job helpers: file counts, the resume check and SUMMARY.md, on the fixture workload."""

import hashlib
import json
from pathlib import Path

import pytest

from clinrag.config import PlannerConfig
from clinrag.eval import freeze, runner, sol_v2
from clinrag.workload.schema import write_jsonl
from tests.fakes import FakeAnswerer
from tests.test_classifier import workload  # noqa: F401  (module-scoped fixture workload)

FIXTURES = Path(__file__).parent / "fixtures"


def run_part(base, router, questions, workload, answerer=None):  # noqa: F811
    """One router's run in base/parts/<router>, as the job runs it (results.* plus run.json)."""
    sources = runner.build_sources("fake", FIXTURES, FIXTURES)
    write_jsonl(base / "workload" / "dev.jsonl", workload.dev)
    part = base / "parts" / router
    runner.run_experiment(sources, questions, [router], PlannerConfig(), base / "workload", part, repeats=1,
                          answerer=answerer)
    (part / "run.json").write_text(json.dumps({"llm_dtype": "bf16"}), encoding="utf-8")
    (part / "command.txt").write_text(f"python -m clinrag.eval.runner --routers {router}\n", encoding="utf-8")
    return part


def test_counts_per_class_and_checksum(tmp_path, workload):  # noqa: F811
    path = tmp_path / "test.jsonl"
    write_jsonl(path, workload.test)
    info = sol_v2.counts(path)
    assert info["n"] == len(workload.test)
    assert sum(info["by_class"].values()) == len(workload.test)
    assert info["by_class"]["A"] == sum(q.spec.qclass == "A" for q in workload.test)
    assert info["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert info["sha256"] in sol_v2.format_counts(info)
    assert sol_v2.main(["counts", str(path)]) == 0


def test_complete_part_is_recognized_and_partial_ones_are_not(tmp_path, workload):  # noqa: F811
    questions = tmp_path / "test.jsonl"
    write_jsonl(questions, workload.test)
    part = run_part(tmp_path, "p1", workload.test, workload, FakeAnswerer())
    assert sol_v2.check_part(part, "p1", questions) is None
    assert sol_v2.check_part(part, "p1", questions, answerer="fake", llm_dtype="bf16") is None
    assert sol_v2.main(["complete", "--part", str(part), "--router", "p1", "--questions", str(questions)]) == 0

    # rows from another answerer or dtype, or another router, are not this run
    assert "expected llama" in sol_v2.check_part(part, "p1", questions, answerer="llama")
    assert "expected 4bit" in sol_v2.check_part(part, "p1", questions, llm_dtype="4bit")
    assert sol_v2.check_part(part, "oracle", questions) is not None

    # a run cut off mid-router: rows missing
    path = part / "results.jsonl"
    path.write_text("".join(path.read_text(encoding="utf-8").splitlines(keepends=True)[:-1]), encoding="utf-8")
    assert "p1" in sol_v2.check_part(part, "p1", questions)
    assert sol_v2.main(["complete", "--part", str(part), "--router", "p1", "--questions", str(questions)]) == 1

    # never started, or stopped before results.csv was written
    assert "results.csv" in sol_v2.check_part(tmp_path / "parts" / "nothing", "p1", questions)
    (part / "results.csv").unlink()
    assert "results.csv" in sol_v2.check_part(part, "p1", questions)


def test_summary_has_answer_metrics_per_router_and_class(tmp_path, workload):  # noqa: F811
    questions = tmp_path / "test.jsonl"
    write_jsonl(questions, workload.test)
    perfect = FakeAnswerer({q.spec.qid: json.dumps({"answer": q.gold["answer"]}) for q in workload.test})
    run = tmp_path / "v2"
    for router in ("oracle", "p1"):
        run_part(run / "template-test", router, workload.test, workload, perfect)
        run_part(run / "paraphrase-test", router, workload.test, workload, perfect)
    freeze.merge_parts({r: run / "paraphrase-test" / "parts" / r for r in ("p1", "oracle")}, questions,
                       run / "paraphrase-test")

    path = sol_v2.summary(run, questions, questions, ["oracle", "p1"], ["p1", "oracle"])
    text = path.read_text(encoding="utf-8")
    assert path == run / "SUMMARY.md"
    header = "| " + " | ".join(sol_v2.COLUMNS) + " |"
    assert text.count(header) == 4  # all classes and per class, for each split
    assert "## Template test split" in text and "## Paraphrase test split" in text
    assert "| oracle | all |" in text and "| p1 | A |" in text
    assert "python -m clinrag.eval.runner --routers p1" in text
    assert "answerer: fake" in text
    # every scored answer was right
    rows = [line for line in text.splitlines() if line.startswith("| oracle | all |")]
    assert rows and all("| 1.000 | 0.000 | 0.000 |" in line for line in rows)
    # the template parts are merged next to them
    merged = (run / "template-test" / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(merged) == 2 * len(workload.test)
    assert (run / "template-test" / "results.csv").exists()


def test_summary_refuses_an_incomplete_router(tmp_path, workload):  # noqa: F811
    questions = tmp_path / "test.jsonl"
    write_jsonl(questions, workload.test)
    run = tmp_path / "v2"
    run_part(run / "template-test", "oracle", workload.test, workload)
    run_part(run / "paraphrase-test", "oracle", workload.test[:-1], workload)
    freeze_into = run / "paraphrase-test"
    (freeze_into / "results.jsonl").write_text(
        (freeze_into / "parts" / "oracle" / "results.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="oracle"):
        sol_v2.summary(run, questions, questions, ["oracle"], ["oracle"])


def write_overall(part_dir, router, tokens, accuracy, n=10):
    """A results.csv with a per-class row and the "all" row fig8 reads."""
    path = part_dir / router / "results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("router,qclass,n,mean_tokens,answer_accuracy\n"
                    f"{router},A,{n // 2},1.0,1.0\n"
                    f"{router},all,{n},{tokens},{accuracy}\n", encoding="utf-8")


def test_fig8_reads_the_overall_rows_and_states_the_takeaway(tmp_path):
    run = tmp_path / "v2"
    for split, p2_acc in (("template-test", 0.7), ("paraphrase-test", 0.6)):
        write_overall(run / split / "parts", "p2_tfidf", 700.0, p2_acc)
        write_overall(run / split / "parts", "query_all", 3500.0, 0.3)
    points = sol_v2.overall_points(run / "template-test" / "parts", ["p2_tfidf", "query_all"])
    assert [(p["router"], p["mean_tokens"], p["answer_accuracy"]) for p in points] == [
        ("p2_tfidf", 700.0, 0.7), ("query_all", 3500.0, 0.3)]
    panels = [points, sol_v2.overall_points(run / "paraphrase-test" / "parts", ["p2_tfidf", "query_all"])]
    title = sol_v2.fig8_title(panels)
    assert "answers more questions correctly than query_all" in title
    assert "0.70 vs 0.30 at 80% fewer tokens" in title and "0.60 vs 0.30" in title

    out = tmp_path / "fig8.png"
    assert sol_v2.main(["fig8", "--run", str(run), "--out", str(out), "--template-routers", "p2_tfidf", "query_all",
                        "--paraphrase-routers", "p2_tfidf", "query_all"]) == 0
    assert out.stat().st_size > 0


def test_fig8_refuses_a_run_without_answer_accuracy(tmp_path):
    write_overall(tmp_path, "p1", 700.0, "")
    with pytest.raises(ValueError, match="answer_accuracy"):
        sol_v2.overall_points(tmp_path, ["p1"])


def test_fig8_groups_coinciding_points_under_one_label():
    points = [{"router": r, "mean_tokens": t, "answer_accuracy": a}
              for r, t, a in [("oracle", 693.0, 0.672), ("p1", 701.0, 0.672), ("query_all", 3200.0, 0.316)]]
    groups = sol_v2._label_groups(points, x_tol=100, y_tol=0.02)
    assert [[p["router"] for p in g] for g in groups] == [["oracle", "p1"], ["query_all"]]

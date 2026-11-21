"""The runner's answer step without a model: flags, lazy import, scoring, context budget, reporting."""

import csv
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from clinrag.config import PlannerConfig, load_planner_config
from clinrag.eval import runner
from clinrag.llm.answerer import NullAnswerer
from clinrag.sources.tokens import count_tokens
from clinrag.workload.schema import write_jsonl
from tests.fakes import FakeAnswerer
from tests.test_classifier import workload  # noqa: F401  (module-scoped fixture workload)

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = PlannerConfig()


@pytest.fixture(scope="module")
def sources():
    return runner.build_sources("fake", FIXTURES, FIXTURES)


@pytest.fixture(scope="module")
def questions(workload):  # noqa: F811
    return [*workload.dev, *workload.test]


def first(questions, prefix):
    return next(q for q in questions if q.template_id.startswith(prefix))


def ask(question, sources, answerer, config=CONFIG, router="oracle"):
    return runner.run_question(
        router, runner.build_router(router, sources, config, Path("."), 0.5), question, sources, config,
        repeats=1, answerer=answerer,
    )


# ---- flags and lazy import ---------------------------------------------------------------


def test_answerer_flags_default_to_no_answer_step():
    args = runner.parse_args([])
    assert (args.answerer, args.llm_dtype) == ("none", "bf16")
    assert runner.parse_args(["--answerer", "llama", "--llm-dtype", "4bit"]).llm_dtype == "4bit"
    assert isinstance(runner.build_answerer(), NullAnswerer)
    assert isinstance(runner.build_answerer(args.answerer, args.llm_dtype), NullAnswerer)
    with pytest.raises(ValueError):
        runner.build_answerer("gpt")
    with pytest.raises(ValueError):
        runner.build_answerer("llama", "fp8")  # rejected before the llama module is imported
    with pytest.raises(SystemExit):
        runner.parse_args(["--answerer", "gpt"])


def test_run_question_defaults_to_null_answerer(sources, questions):
    record = ask(first(questions, "A_latest_"), sources, None)
    assert record["answerer"] == "none"
    assert (record["answer"], record["answer_score"], record["parse_failed"]) == (None, None, None)
    assert record["prompt_tokens"] == 0 and record["answer_latency_ms"] == 0.0


def test_default_run_never_imports_torch_transformers_or_llama(workload, tmp_path):  # noqa: F811
    """A fresh interpreter runs the runner with the default answerer and two routers, then lists its modules."""
    write_jsonl(tmp_path / "workload" / "dev.jsonl", workload.dev)
    write_jsonl(tmp_path / "workload" / "test.jsonl", workload.test)
    code = (
        "import sys\n"
        "from clinrag.eval import runner\n"
        "import clinrag.llm.answerer, clinrag.eval.parse\n"
        f"runner.main(['--sources', 'fake', '--tables-dir', {str(FIXTURES)!r}, '--notes-dir', {str(FIXTURES)!r},\n"
        f"             '--workload', {str(tmp_path / 'workload')!r}, '--split', 'test', '--repeats', '1',\n"
        f"             '--routers', 'p1', 'p2_tfidf', '--no-analysis', '--out', {str(tmp_path / 'run')!r}])\n"
        "loaded = [m for m in ('torch', 'transformers', 'bitsandbytes', 'accelerate', 'clinrag.llm.llama') if m in sys.modules]\n"
        "print('LOADED', loaded)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr
    assert "LOADED []" in done.stdout, done.stdout


def test_llama_module_imports_without_torch():
    code = (
        "import sys\n"
        "import clinrag.llm.llama\n"
        "print('LOADED', [m for m in ('torch', 'transformers') if m in sys.modules])\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr
    assert "LOADED []" in done.stdout, done.stdout


# ---- parsing and scoring in the runner ---------------------------------------------------


def test_fake_answerer_output_is_parsed_and_scored(sources, questions):
    latest = first(questions, "A_latest_")
    ever = first(questions, "A_ever_above_")
    listing = first(questions, "B_list_")
    answerer = FakeAnswerer({
        latest.spec.qid: f'The latest value is {{"answer": "{latest.gold["answer"]} mg/dL"}}',
        ever.spec.qid: "Yes." if ever.gold["answer"] else "No.",
        listing.spec.qid: json.dumps({"answer": [f"Admission {h}" for h in listing.gold["answer"]]}),
    })
    for question in (latest, ever, listing):
        record = ask(question, sources, answerer)
        assert record["answerer"] == "fake"
        assert record["answer_score"] == 1.0, question.spec.text
        assert record["parse_failed"] is False
        assert record["answer_raw"] == answerer.outputs[question.spec.qid]
    assert ask(latest, sources, answerer)["answer"] == pytest.approx(latest.gold["answer"])
    assert ask(listing, sources, answerer)["answer"] == listing.gold["answer"]
    # the answerer was told the template, and got the retrieved evidence
    qid, template_id, evidence = answerer.calls[0]
    assert (qid, template_id) == (latest.spec.qid, latest.template_id)
    assert evidence and all(item["source"] == "relational" for item in evidence)


@pytest.mark.parametrize("output, parse_failed", [
    ('{"answer": null}', False),  # the model declined
    (None, False),  # no answer produced
    ("I am not sure.", True),  # unparseable
    ("", True),
    ('{"answer": "high"}', True),
])
def test_null_or_unparseable_scores_zero_with_a_real_answerer(sources, questions, output, parse_failed):
    for prefix in ("A_latest_", "A_ever_above_", "B_admission_", "B_list_", "C_", "D_"):
        record = ask(first(questions, prefix), sources, FakeAnswerer(default=output))
        assert record["answer"] is None
        assert record["answer_score"] == 0.0 and record["answer_score"] is not None
        assert record["parse_failed"] is parse_failed


def test_wrong_answer_scores_zero_and_is_not_a_parse_failure(sources, questions):
    record = ask(first(questions, "A_max_"), sources, FakeAnswerer(default='{"answer": -1}'))
    assert (record["answer"], record["answer_score"], record["parse_failed"]) == (-1.0, 0.0, False)


# ---- context budget ----------------------------------------------------------------------


def items(*token_counts):
    return [{"hadm_id": i, "text": " ".join(["w"] * 10 * n), "source": "notes"} for i, n in enumerate(token_counts)]


def test_fit_context_fills_in_plan_order_until_the_budget():
    evidence = items(1, 2, 3)  # 13, 26 and 39 tokens
    assert [count_tokens(i["text"]) for i in evidence] == [13, 26, 39]
    assert runner.fit_context(evidence, 78) == (evidence, 78, False)  # exactly the budget fits
    assert runner.fit_context(evidence, 77) == (evidence[:2], 39, True)
    assert runner.fit_context(evidence, 39) == (evidence[:2], 39, True)
    assert runner.fit_context(evidence, 12) == ([], 0, True)
    assert runner.fit_context([], 10) == ([], 0, False)
    # the first item that does not fit ends the context: a later, smaller item is not pulled forward
    assert runner.fit_context(items(1, 3, 1), 30) == (items(1, 3, 1)[:1], 13, True)


def test_default_budget_and_config_file(tmp_path):
    assert PlannerConfig().max_context_tokens == 8192
    path = tmp_path / "config.yaml"
    path.write_text("max_context_tokens: 4096\ntop_k: 3\n", encoding="utf-8")
    assert load_planner_config(path).max_context_tokens == 4096
    with pytest.raises(ValueError):
        PlannerConfig(max_context_tokens=0)


def test_budget_truncates_the_evidence_sent_not_the_retrieval(sources, questions):
    question = first(questions, "C_")
    full_answerer = FakeAnswerer()
    full = ask(question, sources, full_answerer, router="query_all")
    sent_full = full_answerer.calls[0][2]
    assert len(sent_full) > 1 and full["truncated"] is False
    assert full["prompt_tokens_sent"] == full["tokens"] == sum(count_tokens(i["text"]) for i in sent_full)

    budget = full["tokens"] - 1
    small_answerer = FakeAnswerer()
    small = ask(question, sources, small_answerer, replace(CONFIG, max_context_tokens=budget), router="query_all")
    sent_small = small_answerer.calls[0][2]
    assert small["truncated"] is True
    assert sent_small == sent_full[: len(sent_small)] and len(sent_small) < len(sent_full)  # a prefix, in plan order
    assert small["prompt_tokens_sent"] == sum(count_tokens(i["text"]) for i in sent_small) <= budget
    assert small["prompt_tokens"] == small["prompt_tokens_sent"]  # what the fake answerer counted
    # retrieval metrics do not depend on the budget
    assert (small["tokens"], small["recall"], small["plan_name"]) == (full["tokens"], full["recall"], full["plan_name"])


def test_budget_is_the_same_for_every_router(sources, questions):
    question = first(questions, "C_")
    config = replace(CONFIG, max_context_tokens=40)
    for router in ("oracle", "p1", "query_all", "always_filter_first", "always_search_first"):
        answerer = FakeAnswerer()
        record = ask(question, sources, answerer, config, router=router)
        sent = answerer.calls[0][2]
        assert record["prompt_tokens_sent"] == sum(count_tokens(i["text"]) for i in sent) <= 40
        assert record["truncated"] is (record["prompt_tokens_sent"] < record["tokens"])


def test_budget_fields_are_recorded_without_an_answerer(sources, questions):
    record = ask(first(questions, "B_list_"), sources, None)
    assert record["truncated"] is False and record["prompt_tokens_sent"] == record["tokens"]


# ---- reporting ---------------------------------------------------------------------------


def run(tmp_path, workload, answerer, config=CONFIG):  # noqa: F811
    sources = runner.build_sources("fake", FIXTURES, FIXTURES)
    write_jsonl(tmp_path / "workload" / "dev.jsonl", workload.dev)
    out = tmp_path / "run"
    records = runner.run_experiment(
        sources, workload.test, ["oracle", "query_all"], config, tmp_path / "workload", out, repeats=1, answerer=answerer
    )
    with (out / "results.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return records, rows, (out / "results.md").read_text(encoding="utf-8")


def test_summaries_gain_answer_columns_with_an_answerer(tmp_path, workload):  # noqa: F811
    perfect = {q.spec.qid: json.dumps({"answer": q.gold["answer"]}) for q in workload.test}
    class_a = [q.spec.qid for q in workload.test if q.spec.qclass == "A"]
    perfect[class_a[0]] = "no idea"  # one unparseable class A answer
    records, rows, md = run(tmp_path, workload, FakeAnswerer(perfect))

    assert set(rows[0]) == set(runner.SUMMARY_FIELDS) | set(runner.ANSWER_FIELDS)
    by_key = {(r["router"], r["qclass"]): r for r in rows}
    n_a, n = len(class_a), len(workload.test)
    for router in ("oracle", "query_all"):
        a, overall = by_key[router, "A"], by_key[router, "all"]
        assert float(a["answer_accuracy"]) == pytest.approx((n_a - 1) / n_a)
        assert float(a["parse_failure_rate"]) == pytest.approx(1 / n_a)
        assert float(overall["answer_accuracy"]) == pytest.approx((n - 1) / n)
        assert float(overall["parse_failure_rate"]) == pytest.approx(1 / n)
        assert float(overall["truncated_rate"]) == 0.0
        for qclass in ("B", "C", "D"):
            if (router, qclass) in by_key:
                assert float(by_key[router, qclass]["answer_accuracy"]) == 1.0
                assert float(by_key[router, qclass]["parse_failure_rate"]) == 0.0
    for field in runner.ANSWER_FIELDS:
        assert f"| {field} |" in md or f" {field} |" in md
        assert f"{field}:" in md
    assert all(r["answerer"] == "fake" and r["answer_score"] is not None for r in records)


def test_truncated_rate_is_reported(tmp_path, workload):  # noqa: F811
    _, rows, _ = run(tmp_path, workload, FakeAnswerer(), replace(CONFIG, max_context_tokens=1))
    by_key = {(r["router"], r["qclass"]): r for r in rows}
    assert float(by_key["query_all", "all"]["truncated_rate"]) == 1.0  # every question retrieved something
    assert float(by_key["query_all", "all"]["answer_accuracy"]) == 0.0  # null answers score 0, not missing


def test_summaries_have_no_answer_columns_without_an_answerer(tmp_path, workload):  # noqa: F811
    records, rows, md = run(tmp_path, workload, None)
    assert set(rows[0]) == set(runner.SUMMARY_FIELDS)
    assert not any(field in md for field in runner.ANSWER_FIELDS)
    assert all(r["answerer"] == "none" and r["answer_score"] is None for r in records)

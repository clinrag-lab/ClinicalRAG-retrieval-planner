"""Scaled synthetic notes and the order check across index sizes, on the fixture."""

import json
from pathlib import Path

import pandas as pd
import pytest

from clinrag.eval import scale_study
from clinrag.workload import notes_gen
from clinrag.workload.notes_gen import generate_notes, mentioned_concepts
from clinrag.workload.schema import write_jsonl
from tests.test_classifier import workload  # noqa: F401  (module-scoped fixture workload)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def notes():
    return generate_notes(FIXTURES, seed=0), generate_notes(FIXTURES, seed=0, scale=3)


def test_scale_writes_s_notes_per_admission(notes):
    base, scaled = notes
    d = scaled.discharge
    assert len(d) == 3 * len(base.discharge)
    assert all(sorted(seqs) == [1, 2, 3] for seqs in d.groupby("hadm_id")["note_seq"].apply(list))
    assert d["note_id"].is_unique
    # Note 1 of each admission is the scale-1 note.
    first = d[d["note_seq"] == 1].set_index("hadm_id")["text"]
    assert first.to_dict() == base.discharge.set_index("hadm_id")["text"].to_dict()
    # Later notes are re-phrased.
    assert (d[d["note_seq"] == 2]["text"].values != d[d["note_seq"] == 1]["text"].values).any()


def test_scaled_notes_keep_per_admission_concepts(notes):
    base, scaled = notes
    for frame_name in ("concepts", "negations"):
        before = set(map(tuple, getattr(base, frame_name)[["hadm_id", "concept"]].values.tolist()))
        after = set(map(tuple, getattr(scaled, frame_name)[["hadm_id", "concept"]].values.tolist()))
        assert before == after
    # Every scaled note mentions exactly its admission's tagged and negated concepts.
    expected = {}
    for frame in (base.concepts, base.negations):
        for hadm, concept in frame[["hadm_id", "concept"]].values.tolist():
            expected.setdefault(hadm, set()).add(concept)
    for hadm, text in scaled.discharge[["hadm_id", "text"]].values.tolist():
        assert mentioned_concepts(text) == expected.get(hadm, set())


def test_scale_is_deterministic(notes):
    _, scaled = notes
    again = generate_notes(FIXTURES, seed=0, scale=3)
    pd.testing.assert_frame_equal(scaled.discharge, again.discharge)
    with pytest.raises(ValueError):
        generate_notes(FIXTURES, seed=0, scale=0)


def test_cli_writes_scaled_notes_to_their_own_folder(tmp_path):
    notes_gen.main(["--tables-dir", str(FIXTURES), "--out-dir", str(tmp_path), "--seed", "0", "--scale", "2"])
    assert (tmp_path / "scale_2" / "discharge.csv.gz").exists()
    assert not (tmp_path / "discharge.csv.gz").exists()


def test_scale_study_end_to_end(workload, tmp_path):  # noqa: F811
    root = tmp_path / "generated"
    notes_gen.main(["--tables-dir", str(FIXTURES), "--out-dir", str(root), "--seed", "0"])
    notes_gen.main(["--tables-dir", str(FIXTURES), "--out-dir", str(root), "--seed", "0", "--scale", "4"])
    write_jsonl(tmp_path / "wl" / "dev.jsonl", workload.dev)
    write_jsonl(tmp_path / "wl" / "test.jsonl", workload.test)
    out = tmp_path / "run"
    summaries = scale_study.main([
        "--tables-dir", str(FIXTURES), "--notes-root", str(root), "--workload", str(tmp_path / "wl"),
        "--split", "test", "--scales", "1", "4", "--config", str(tmp_path / "missing.yaml"),
        "--repeats", "1", "--out", str(out),
    ])
    assert [s["total_notes"] for s in summaries] == [8, 32]
    assert all(s["n"] > 0 for s in summaries)
    # The rule switches at the same matching-note count, a smaller selectivity on the larger index.
    assert summaries[1]["switch_selectivity"] == pytest.approx(summaries[0]["switch_selectivity"] / 4)
    for name in ("fig2_order_vs_selectivity.png", "fig5_scale.png"):
        assert (out / "figures" / name).read_bytes()[:4] == b"\x89PNG"
    data = json.loads((out / "analysis" / "order_scale_summary.json").read_text(encoding="utf-8"))
    assert data["split"] == "test" and len(data["summaries"]) == 2
    assert (out / "analysis" / "calibrated_config_scale_4.yaml").exists()
    assert (out / "analysis" / "order_check_scale_4.csv").exists()

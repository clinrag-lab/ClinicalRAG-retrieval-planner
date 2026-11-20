from pathlib import Path

import pandas as pd
import pytest

from clinrag.workload.icd_categories import CATEGORIES, DIABETES, FALL, RENAL_FAILURE, SEPSIS
from clinrag.workload.notes_gen import (
    CONCEPT_PATTERNS,
    CONCEPTS,
    FILLER_SENTENCES,
    MAX_WORDS,
    MIN_WORDS,
    NEGATED_PHRASES,
    POSITIVE_PHRASES,
    concept_probability,
    generate_notes,
    main,
    mentioned_concepts,
)

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_HADMS = list(range(20000001, 20000009))


@pytest.fixture(scope="module")
def notes():
    return generate_notes(FIXTURES, seed=0)


def test_deterministic_for_fixed_seed(notes):
    again = generate_notes(FIXTURES, seed=0)
    pd.testing.assert_frame_equal(notes.discharge, again.discharge)
    pd.testing.assert_frame_equal(notes.concepts, again.concepts)
    pd.testing.assert_frame_equal(notes.negations, again.negations)


def test_seed_changes_text(notes):
    other = generate_notes(FIXTURES, seed=1)
    assert list(notes.discharge["text"]) != list(other.discharge["text"])


def test_one_note_per_admission(notes):
    d = notes.discharge
    assert sorted(d["hadm_id"]) == FIXTURE_HADMS
    assert d["note_id"].is_unique
    assert set(d["note_type"]) == {"DS"}
    assert list(d.columns) == ["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "text"]


def test_note_ids_follow_mimic_format(notes):
    # Subject 10000001 has two admissions: note_seq 1 and 2 in admission order.
    rows = notes.discharge[notes.discharge["subject_id"] == 10000001]
    assert list(rows["note_id"]) == ["10000001-DS-1", "10000001-DS-2"]
    assert list(rows["hadm_id"]) == [20000001, 20000002]


def test_charttime_is_discharge_time(notes):
    row = notes.discharge.set_index("hadm_id").loc[20000001]
    assert row["charttime"] == "2150-03-06 11:00:00"


def test_word_counts_in_range(notes):
    words = notes.discharge["text"].str.split().str.len()
    assert words.between(MIN_WORDS, MAX_WORDS).all()


def test_every_tagged_concept_appears_in_text(notes):
    texts = notes.discharge.set_index("note_id")["text"]
    for _, row in notes.concepts.iterrows():
        assert CONCEPT_PATTERNS[row["concept"]].search(texts[row["note_id"]]), row.to_dict()


def test_negated_mentions_are_never_tagged(notes):
    tagged = set(zip(notes.concepts["note_id"], notes.concepts["concept"]))
    negated = set(zip(notes.negations["note_id"], notes.negations["concept"]))
    assert not tagged & negated
    texts = notes.discharge.set_index("note_id")["text"]
    for note_id, concept in negated:
        assert any(p in texts[note_id] for p in NEGATED_PHRASES[concept])


def test_text_mentions_exactly_tagged_plus_negated(notes):
    for _, row in notes.discharge.iterrows():
        tagged = set(notes.concepts.loc[notes.concepts["note_id"] == row["note_id"], "concept"])
        negated = set(notes.negations.loc[notes.negations["note_id"] == row["note_id"], "concept"])
        assert mentioned_concepts(row["text"]) == tagged | negated


def test_concept_in_diagnosis_title_is_always_tagged():
    # Fixture admission 20000006 has diagnosis R079 "Chest pain, unspecified".
    for seed in range(5):
        concepts = generate_notes(FIXTURES, seed=seed).concepts
        assert "chest pain" in set(concepts.loc[concepts["hadm_id"] == 20000006, "concept"])


def test_phrases_mention_only_their_concept():
    for concept in CONCEPTS:
        for phrase in POSITIVE_PHRASES[concept] + NEGATED_PHRASES[concept]:
            assert mentioned_concepts(phrase) == {concept}, phrase
    for sentence in FILLER_SENTENCES:
        assert mentioned_concepts(sentence) == set(), sentence


def test_concept_probability_is_noisy_or():
    # dialysis: base 0.02, renal failure 0.50, creatinine > 2 0.35
    assert concept_probability("dialysis", set()) == pytest.approx(0.02)
    assert concept_probability("dialysis", {"renal_failure_dx", "creatinine_above_2"}) == pytest.approx(
        1 - 0.98 * 0.50 * 0.65
    )
    assert concept_probability("dialysis", {"diabetes_dx"}) == pytest.approx(0.02)


@pytest.mark.parametrize(
    "group,code,version,expected",
    [
        (CATEGORIES["circulatory"], "I214", 10, True),
        (CATEGORIES["circulatory"], "4019", 9, True),
        (CATEGORIES["circulatory"], "V4581", 9, False),
        (CATEGORIES["renal_genitourinary"], "N186", 10, True),
        (CATEGORIES["renal_genitourinary"], "5849", 9, True),
        (CATEGORIES["respiratory"], "J189", 10, True),
        (CATEGORIES["respiratory"], "486", 9, True),
        (CATEGORIES["endocrine"], "E119", 10, True),
        (CATEGORIES["endocrine"], "E8800", 9, False),  # ICD-9 E code is an external cause, not endocrine
        (RENAL_FAILURE, "N179", 10, True),
        (RENAL_FAILURE, "N390", 10, False),
        (RENAL_FAILURE, "5856", 9, True),
        (DIABETES, "25000", 9, True),
        (DIABETES, "E119", 10, True),
        (SEPSIS, "A419", 10, True),
        (SEPSIS, "99591", 9, True),
        (SEPSIS, "0389", 9, True),
        (FALL, "W1830XA", 10, True),
        (FALL, "E8889", 9, True),
    ],
)
def test_icd_categories(group, code, version, expected):
    assert group.contains(code, version) is expected


def test_cli_writes_files(tmp_path):
    main(["--tables-dir", str(FIXTURES), "--out-dir", str(tmp_path), "--seed", "0"])
    for name in ("discharge.csv.gz", "note_concepts.csv.gz", "note_negations.csv.gz"):
        assert (tmp_path / name).exists()
    assert len(pd.read_csv(tmp_path / "discharge.csv.gz")) == 8

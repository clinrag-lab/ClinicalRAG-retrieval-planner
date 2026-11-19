import pytest

from clinrag.eval.metrics import normalized_recall_at_k, numeric_match, score_answer, set_f1, yes_no_match
from clinrag.question import QuestionSpec, StructuredFilter
from clinrag.workload.schema import WorkloadQuestion

# ---- normalized recall@k ----------------------------------------------------------


def test_recall_all_gold_found():
    # gold {1, 2}, top 3 = [1, 9, 2] -> 2 hits / min(3, 2) = 1.0
    assert normalized_recall_at_k([1, 9, 2, 3], [1, 2], k=3) == 1.0


def test_recall_partial():
    # gold {1, 2, 3, 4}, top 4 = [1, 9, 8, 3] -> 2 hits / min(4, 4) = 0.5
    assert normalized_recall_at_k([1, 9, 8, 3], [1, 2, 3, 4], k=4) == 0.5


def test_recall_gold_larger_than_k():
    # 10 gold admissions, k = 5. Perfect top 5 -> 5 / min(5, 10) = 1.0
    gold = list(range(1, 11))
    assert normalized_recall_at_k([1, 2, 3, 4, 5, 6], gold, k=5) == 1.0
    # 3 of the top 5 are gold -> 3 / 5 = 0.6; gold beyond rank 5 does not count
    assert normalized_recall_at_k([1, 99, 2, 98, 3, 4, 5], gold, k=5) == pytest.approx(0.6)


def test_recall_duplicate_admissions_count_once():
    # Two chunks from admission 1 in the top 3: hits {1, 2} -> 2 / min(3, 3)
    assert normalized_recall_at_k([1, 1, 2], [1, 2, 3], k=3) == pytest.approx(2 / 3)


def test_recall_empty_gold():
    assert normalized_recall_at_k([], [], k=5) == 1.0
    assert normalized_recall_at_k([4], [], k=5) == 0.0


def test_recall_nothing_retrieved():
    assert normalized_recall_at_k([], [1, 2], k=5) == 0.0


def test_recall_rejects_bad_k():
    with pytest.raises(ValueError):
        normalized_recall_at_k([1], [1], k=0)


# ---- set F1, numeric, yes/no ------------------------------------------------------


def test_set_f1():
    # predicted {1, 2, 3}, gold {2, 3, 4, 5}: P = 2/3, R = 2/4, F1 = 2PR/(P+R) = 4/7
    assert set_f1([1, 2, 3], [2, 3, 4, 5]) == pytest.approx(4 / 7)
    assert set_f1([2, 3], [3, 2]) == 1.0
    assert set_f1([1], [2]) == 0.0
    assert set_f1([], []) == 1.0
    assert set_f1([], [1]) == 0.0
    assert set_f1([1, 1, 2], [1, 2]) == 1.0


def test_numeric_match():
    assert numeric_match(2.9, 2.9)
    assert numeric_match(2.92, 2.9)  # 0.69% off
    assert not numeric_match(2.95, 2.9)  # 1.7% off
    assert numeric_match("4.9", 4.9)
    assert numeric_match(4.95, 4.9, rel_tol=0.02)
    assert not numeric_match("about five", 4.9)
    assert not numeric_match(None, 4.9)
    assert not numeric_match(True, 1.0)


@pytest.mark.parametrize(
    "predicted,gold,expected",
    [(1, 1, True), (0, 0, True), (1, 0, False), ("Yes", 1, True), ("no.", 0, True), (True, 1, True), ("maybe", 1, False), (2, 1, False)],
)
def test_yes_no_match(predicted, gold, expected):
    assert yes_no_match(predicted, gold) is expected


# ---- score_answer dispatch --------------------------------------------------------


def wq(qclass, template_id, answer, **spec):
    sources = {"A": {"relational"}, "B": {"notes"}, "C": {"relational", "notes"}, "D": {"graph", "relational", "notes"}}
    q = QuestionSpec(qid="q", text="t", qclass=qclass, required_sources=frozenset(sources[qclass]), **spec)
    return WorkloadQuestion(q, {"answer": answer, "template_id": template_id})


LAB = StructuredFilter(50912, hadm_id=1)


def test_score_class_a_numeric():
    q = wq("A", "A_latest_creatinine", 2.9, relational_filter=LAB)
    assert score_answer(q, 2.91).metric == "numeric_match"
    assert score_answer(q, 2.91).value == 1.0
    assert score_answer(q, 3.5).value == 0.0


def test_score_class_a_yes_no():
    q = wq("A", "A_ever_above_creatinine", 1, relational_filter=StructuredFilter(50912, ">", 2.0, hadm_id=1))
    assert score_answer(q, "yes").metric == "yes_no_match"
    assert score_answer(q, "yes").value == 1.0
    assert score_answer(q, 0).value == 0.0


def test_score_class_b_admission_yes_no():
    q = wq("B", "B_admission_dialysis", 0, note_query="dialysis", note_hadm_id=7)
    assert score_answer(q, "no").metric == "yes_no_match"
    assert score_answer(q, "no").value == 1.0


def test_score_list_classes_use_set_f1():
    b = wq("B", "B_list_dialysis", [1, 2], note_query="dialysis")
    c = wq("C", "C_creatinine_dialysis", [2, 3, 4, 5], relational_filter=StructuredFilter(50912, ">", 2.0), note_query="dialysis")
    d = wq("D", "D_renal_genitourinary_dialysis", [5], note_query="dialysis", category="renal_genitourinary")
    assert score_answer(b, [1, 2]).value == 1.0
    assert score_answer(c, [1, 2, 3]).metric == "set_f1"
    assert score_answer(c, [1, 2, 3]).value == pytest.approx(4 / 7)
    assert score_answer(d, []).value == 0.0
    with pytest.raises(TypeError):
        score_answer(d, "20000001")

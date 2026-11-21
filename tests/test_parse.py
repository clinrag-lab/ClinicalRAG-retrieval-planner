"""Shared answer parsing: first JSON object anywhere, bare-value fallback, type coercion."""

import pytest

from clinrag.eval.parse import ParsedAnswer, answer_type, first_json_object, parse_answer, score_parsed
from clinrag.question import QuestionSpec, StructuredFilter
from clinrag.workload.schema import WorkloadQuestion

OK = lambda value: ParsedAnswer(value, False)  # noqa: E731
DECLINED = ParsedAnswer(None, False)
FAILED = ParsedAnswer(None, True)


@pytest.mark.parametrize("template_id, expected", [
    ("A_latest_creatinine", "number"),
    ("A_max_potassium", "number"),
    ("A_ever_above_creatinine", "yesno"),
    ("B_admission_chest_pain", "yesno"),
    ("B_list_dialysis", "list"),
    ("C_creatinine_dialysis", "list"),
    ("D_heart_failure_sepsis", "list"),
])
def test_answer_type_comes_from_the_template(template_id, expected):
    assert answer_type(template_id) == expected


def test_answer_type_rejects_unknown_templates():
    with pytest.raises(ValueError):
        answer_type("E_something")
    with pytest.raises(ValueError):
        parse_answer("1", "text")


@pytest.mark.parametrize("output, obj", [
    ('{"answer": 4.9}', {"answer": 4.9}),
    ('Sure! Here is the JSON:\n{"answer": 4.9}\nHope that helps.', {"answer": 4.9}),
    ('```json\n{"answer": [1, 2]}\n```', {"answer": [1, 2]}),
    ('{"answer": 1} and later {"answer": 2}', {"answer": 1}),  # the first object wins
    ('{not json} then {"answer": "yes"}', {"answer": "yes"}),  # skips a brace that does not decode
    ('{"answer": {"value": 3}} trailing', {"answer": {"value": 3}}),
    ("no braces here", None),
    ("{broken", None),
])
def test_first_json_object_anywhere(output, obj):
    assert first_json_object(output) == obj


@pytest.mark.parametrize("output, value", [
    ('{"answer": 4.9}', 4.9),
    ('{"answer": 5}', 5.0),
    ('The answer is {"answer": 4.9}.', 4.9),
    ('{"answer": "4.9 mg/dL"}', 4.9),  # units stripped
    ('{"answer": "1,250.5 mg/dL"}', 1250.5),  # commas stripped
    ('{"answer": "12,345"}', 12345.0),
    ('{"answer": "-0.5"}', -0.5),
    ("4.9", 4.9),  # bare value
    ("4.9 mg/dL", 4.9),
    ("The most recent value was 1,250 mg/dL.", 1250.0),
    ("  7  ", 7.0),
])
def test_numbers(output, value):
    assert parse_answer(output, "number") == OK(pytest.approx(value))


@pytest.mark.parametrize("output, value", [
    ('{"answer": "yes"}', 1),
    ('{"answer": "no"}', 0),
    ('{"answer": "Yes."}', 1),  # case and punctuation
    ('{"answer": "NO!"}', 0),
    ('{"answer": true}', 1),
    ('{"answer": false}', 0),
    ('{"answer": 1}', 1),
    ('{"answer": 0}', 0),
    ('{"answer": "No, it never exceeded the threshold."}', 0),
    ("Yes", 1),  # bare values
    ("no.", 0),
    ("YES!", 1),
    ("  No  ", 0),
    ("y", 1),
    ("N.", 0),
    ("true", 1),
    ("1", 1),
    ("Answer: yes, the summary mentions dialysis.", 1),
])
def test_yes_no(output, value):
    assert parse_answer(output, "yesno") == OK(value)


@pytest.mark.parametrize("output, value", [
    ('{"answer": [20000001, 20000002]}', [20000001, 20000002]),
    ('{"answer": []}', []),  # an empty list is an answer, not a failure
    ('{"answer": ["Admission 123", "Admission 456"]}', [123, 456]),
    ('{"answer": ["123", 456.0, "hadm_id 789"]}', [123, 456, 789]),
    ('{"answer": "Admission 123 and Admission 456"}', [123, 456]),
    ('{"answer": [123, 123, 456]}', [123, 456]),  # duplicates dropped, order kept
    ('{"answer": 123}', [123]),
    ('Here you go: {"answer": ["Admission 123"]} (from the notes)', [123]),
    ("[123, 456]", [123, 456]),  # bare values
    ("The admissions are [123, 456].", [123, 456]),
    ("Admission 123, Admission 456", [123, 456]),
])
def test_lists(output, value):
    assert parse_answer(output, "list") == OK(value)


@pytest.mark.parametrize("output, expected", [
    ('{"answer": null}', "number"),
    ('{"answer": null}', "yesno"),
    ('{"answer": null}', "list"),
    ('Not enough evidence. {"answer": null}', "number"),
    ("null", "number"),
    ("None.", "list"),
    ('{"answer": "unknown"}', "yesno"),
    (None, "number"),
])
def test_null_is_declined_not_a_parse_failure(output, expected):
    assert parse_answer(output, expected) == DECLINED


@pytest.mark.parametrize("output, expected", [
    ("", "number"),
    ("   ", "list"),
    ("I cannot tell from the evidence.", "number"),
    ('{"answer": "high"}', "number"),
    ('{"answer": true}', "number"),
    ('{"answer": [1, 2]}', "number"),
    ('{"value": 4.9}', "number"),  # no "answer" key and the text is not a bare number
    ("maybe", "yesno"),
    ('{"answer": 2}', "yesno"),
    ('{"answer": [1]}', "yesno"),
    ("There are several admissions.", "list"),
    ('{"answer": ["none of them"]}', "list"),
    ('{"answer": [true]}', "list"),
    ('{"answer": [1.5]}', "list"),
    ('{"answer": {"ids": [1]}}', "list"),
])
def test_unparseable_output_fails(output, expected):
    assert parse_answer(output, expected) == FAILED


def test_values_pass_through_for_answerers_that_return_them():
    assert parse_answer(4.9, "number") == OK(4.9)
    assert parse_answer(True, "yesno") == OK(1)
    assert parse_answer([1, 2], "list") == OK([1, 2])
    assert parse_answer([1, 2], "number") == FAILED


def wq(template_id, gold, qclass="A"):
    spec = QuestionSpec(qid="q", text="t", qclass=qclass, relational_filter=StructuredFilter(50912, hadm_id=1),
                        required_sources=frozenset({"relational"}))
    return WorkloadQuestion(spec, {"answer": gold, "template_id": template_id})


def test_score_parsed_scores_null_and_unparseable_as_zero():
    latest = wq("A_latest_creatinine", 4.9)
    assert score_parsed(latest, parse_answer('{"answer": "4.9 mg/dL"}', "number")) == 1.0
    assert score_parsed(latest, parse_answer('{"answer": 6.4}', "number")) == 0.0
    assert score_parsed(latest, DECLINED) == 0.0
    assert score_parsed(latest, FAILED) == 0.0
    ever = wq("A_ever_above_creatinine", 0)
    assert score_parsed(ever, parse_answer("No.", "yesno")) == 1.0  # a parsed 0 is a value, not a missing answer
    assert score_parsed(ever, DECLINED) == 0.0

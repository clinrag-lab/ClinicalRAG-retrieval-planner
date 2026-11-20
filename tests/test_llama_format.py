"""Llama answerer prompt pieces that need no model: evidence text and the expected answer type.

LlamaAnswerer objects are made with __new__, so no tokenizer or weights are loaded.
"""

import inspect
from pathlib import Path

import pytest

from clinrag.llm import llama
from clinrag.llm.llama import LAB_ORDER_NOTE, LlamaAnswerer, format_evidence, lab_line
from clinrag.question import QuestionSpec, StructuredFilter
from clinrag.sources.base import RelationalQuery
from clinrag.sources.reference import build_reference_sources

FIXTURES = Path(__file__).parent / "fixtures"
CREATININE = 50912


def lab(hadm_id, charttime, value, label="Creatinine", unit="mg/dL"):
    return {"hadm_id": hadm_id, "itemid": CREATININE, "label": label, "valuenum": value, "unit": unit,
            "charttime": charttime, "text": f"Admission {hadm_id}: {label} {value} {unit} at {charttime}",
            "source": "relational"}


def note(hadm_id, text):
    return {"hadm_id": hadm_id, "note_id": f"n{hadm_id}", "text": text, "score": 0.5, "source": "notes"}


def test_lab_row_is_one_line():
    assert lab_line(lab(2, "2151-07-18 06:00:00", 4.9)) == "2151-07-18 06:00:00 | Creatinine | 4.9 mg/dL"
    assert lab_line(lab(2, "2151-07-18 06:00:00", 12.0, "Hemoglobin", "g/dL")) == "2151-07-18 06:00:00 | Hemoglobin | 12 g/dL"
    assert lab_line(lab(2, "2151-07-18 06:00:00", 4.9, unit="")) == "2151-07-18 06:00:00 | Creatinine | 4.9"
    # rows from other sources may lack the label or time
    assert lab_line({"hadm_id": 2, "itemid": CREATININE, "valuenum": 1.2}) == "unknown time | item 50912 | 1.2"


def test_lab_rows_are_sorted_oldest_to_newest_with_the_note():
    rows = [lab(2, "2151-07-18 06:00:00", 4.9), lab(2, "2151-07-15 10:00:00", 6.4), lab(2, "2151-07-16 08:30:00", 5.5)]
    assert format_evidence(rows) == "\n".join([
        "Lab rows are listed oldest to newest; the last row is the most recent.",
        "",
        "Admission 2 lab rows:",
        "2151-07-15 10:00:00 | Creatinine | 6.4 mg/dL",
        "2151-07-16 08:30:00 | Creatinine | 5.5 mg/dL",
        "2151-07-18 06:00:00 | Creatinine | 4.9 mg/dL",
    ])
    assert LAB_ORDER_NOTE == "Lab rows are listed oldest to newest; the last row is the most recent."
    assert format_evidence(rows).splitlines()[-1].endswith("| 4.9 mg/dL")  # the last row is the most recent


def test_rows_are_grouped_by_admission_in_retrieval_order():
    text = format_evidence([lab(7, "2150-01-02 00:00:00", 3.0), lab(2, "2151-07-18 06:00:00", 4.9),
                            lab(7, "2150-01-01 00:00:00", 2.0)])
    assert text.split("\n\n") == [
        LAB_ORDER_NOTE,
        "Admission 7 lab rows:\n2150-01-01 00:00:00 | Creatinine | 2 mg/dL\n2150-01-02 00:00:00 | Creatinine | 3 mg/dL",
        "Admission 2 lab rows:\n2151-07-18 06:00:00 | Creatinine | 4.9 mg/dL",
    ]


def test_notes_stay_as_text_with_their_admission_id():
    text = format_evidence([note(7, "Dialysis was performed.\nPatient tolerated it well."), note(2, "No chest pain.")])
    assert text == (
        "Admission 7 note:\nDialysis was performed.\nPatient tolerated it well.\n\n"
        "Admission 2 note:\nNo chest pain."
    )
    assert LAB_ORDER_NOTE not in text  # no lab rows, no lab note


def test_mixed_evidence_keeps_plan_order():
    diagnosis = {"hadm_id": 3, "icd_code": "N179", "icd_version": 10, "source": "relational",
                 "text": "Admission 3: diagnosis ICD-10 N179 Acute kidney failure"}
    text = format_evidence([diagnosis, lab(2, "2151-07-18 06:00:00", 4.9), note(7, "Dialysis was performed.")])
    assert text.split("\n\n") == [
        LAB_ORDER_NOTE,
        "Admission 3: diagnosis ICD-10 N179 Acute kidney failure",
        "Admission 2 lab rows:\n2151-07-18 06:00:00 | Creatinine | 4.9 mg/dL",
        "Admission 7 note:\nDialysis was performed.",
    ]
    assert format_evidence([]) == "No evidence was retrieved."
    assert "{" not in text  # no JSON dumps of the items


def test_rows_from_the_relational_source_format_cleanly():
    sources = build_reference_sources(FIXTURES, FIXTURES)
    rows = sources.relational.run(RelationalQuery(lab=StructuredFilter(CREATININE, hadm_id=20000002))).items
    assert format_evidence(list(reversed(rows))).splitlines()[2:] == [
        "Admission 20000002 lab rows:",
        "2151-07-15 10:00:00 | Creatinine | 6.4 mg/dL",
        "2151-07-18 06:00:00 | Creatinine | 4.9 mg/dL",
    ]
    one = sources.relational.run(RelationalQuery(lab=StructuredFilter(CREATININE, hadm_id=20000002, agg="latest"))).items
    assert format_evidence(one).splitlines()[-1] == "2151-07-18 06:00:00 | Creatinine | 4.9 mg/dL"


class WordTokenizer:
    """Stands in for the Llama tokenizer: one token per whitespace-separated word."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return " ".join(token_ids)


def answerer(max_evidence_tokens=None):
    made = LlamaAnswerer.__new__(LlamaAnswerer)  # no model
    made.tokenizer = WordTokenizer()
    made.max_evidence_tokens = max_evidence_tokens
    return made


def test_answerer_formats_evidence_and_leaves_the_budget_to_the_caller(capsys):
    rows = [lab(2, "2151-07-18 06:00:00", 4.9), lab(2, "2151-07-15 10:00:00", 6.4)]
    assert answerer(None)._format_evidence(rows) == format_evidence(rows)
    assert capsys.readouterr().out == ""
    capped = answerer(5)._format_evidence(rows)  # its own cap still works when set
    assert capped == "Lab rows are listed oldest"
    assert "truncated" in capsys.readouterr().out


def spec(text, flt=None, **kw):
    sources = {"relational"} if flt is not None else {"notes"}
    return QuestionSpec(qid="A070", text=text, qclass="A" if flt is not None else "B", relational_filter=flt,
                        required_sources=frozenset(sources), **kw)


@pytest.mark.parametrize("template_id, expected", [
    ("A_latest_creatinine", "number"),
    ("A_max_creatinine", "number"),
    ("A_ever_above_creatinine", "yesno"),
    ("B_admission_dialysis", "yesno"),
    ("B_list_dialysis", "list"),
    ("C_creatinine_dialysis", "list"),
    ("D_heart_failure_sepsis", "list"),
])
def test_answer_type_follows_the_template_not_the_text(template_id, expected):
    # the text reads like an "ever above" question whatever the template is
    misleading = spec("Was creatinine ever above 2 mg/dL?", StructuredFilter(CREATININE, hadm_id=1))
    assert answerer()._answer_type(misleading, template_id) == expected


def test_paraphrased_ever_above_is_still_yes_no():
    """Held-out phrasings never say "ever above"; the template id still gives the type."""
    flt = StructuredFilter(CREATININE, ">", 2.0, hadm_id=1)
    paraphrased = spec("Pull up encounter 1: any Cr that topped 2?", flt)
    assert answerer()._answer_type(paraphrased, "A_ever_above_creatinine") == "yesno"
    assert answerer()._answer_type(paraphrased) == "yesno"  # no template: from the filter, not the text
    plain = spec("Was it ever above the others?", StructuredFilter(CREATININE, hadm_id=1, agg="max"))
    assert answerer()._answer_type(plain) == "number"
    assert answerer()._answer_type(spec("was dialysis done?", note_query="dialysis", note_hadm_id=1)) == "yesno"
    assert answerer()._answer_type(spec("who had dialysis?", note_query="dialysis")) == "list"


def test_no_debug_print_for_a_single_question():
    source = inspect.getsource(llama)
    assert "A070" not in source and "=== END EVIDENCE" not in source

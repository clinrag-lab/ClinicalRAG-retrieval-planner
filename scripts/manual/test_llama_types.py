"""Manual check, not part of pytest: loads Llama-3.1-8B-Instruct (GPU and Hugging Face access needed).

    python scripts/manual/test_llama_types.py
"""

from clinrag.eval.parse import parse_answer
from clinrag.llm.llama import LlamaAnswerer
from clinrag.question import QuestionSpec

# Load Llama only once for both tests.
answerer = LlamaAnswerer()


# TEST 1: Yes/No question (Class B)
question_yesno = QuestionSpec(
    qid="test_yesno",
    text=(
        "According to the discharge summary for admission 12345, "
        "was the patient treated with antibiotics?"
    ),
    qclass="B",
    note_query="antibiotics",
    note_hadm_id=12345,
    required_sources=frozenset({"notes"}),
)

evidence_yesno = [
    {
        "source": "notes",
        "hadm_id": 12345,
        "text": (
            "The patient was admitted with pneumonia. "
            "They received intravenous antibiotics during "
            "their hospitalization."
        ),
    }
]

raw, tokens, latency = answerer.answer(
    question_yesno, evidence_yesno, template_id="B_admission_antibiotics"
)
answer = parse_answer(raw, "yesno").value

print("\n=== YES/NO TEST ===")
print("Answer:", answer)
print("Expected:", 1)
print("Correct:", answer == 1)
print("Prompt tokens:", tokens)
print("Latency (ms):", round(latency, 2))


# TEST 2: Admission ID list (Class B)
question_list = QuestionSpec(
    qid="test_list",
    text="Which admissions mention treatment with antibiotics?",
    qclass="B",
    note_query="antibiotics",
    required_sources=frozenset({"notes"}),
)

evidence_list = [
    {
        "source": "notes",
        "hadm_id": 101,
        "text": "The patient received antibiotics for pneumonia.",
    },
    {
        "source": "notes",
        "hadm_id": 102,
        "text": "The patient received IV fluids for dehydration.",
    },
    {
        "source": "notes",
        "hadm_id": 103,
        "text": "The patient completed a course of antibiotics.",
    },
]

raw, tokens, latency = answerer.answer(
    question_list, evidence_list, template_id="B_list_antibiotics"
)
answer = parse_answer(raw, "list").value

print("\n=== LIST TEST ===")
print("Answer:", answer)
print("Expected:", [101, 103])
print("Correct:", set(answer or []) == {101, 103})
print("Prompt tokens:", tokens)
print("Latency (ms):", round(latency, 2))

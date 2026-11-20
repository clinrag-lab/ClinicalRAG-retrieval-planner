
from clinrag.llm.llama import LlamaAnswerer
from clinrag.question import QuestionSpec, StructuredFilter

# Create a test clinical question.
question = QuestionSpec(
    qid="test_001",
    text="What is the maximum creatinine value for this admission?",
    qclass="A",
    relational_filter=StructuredFilter(
        itemid=50912,
        hadm_id=12345,
    ),
    required_sources=frozenset({"relational"}),
)

# Synthetic evidence returned by retrieval.
evidence = [
    {
        "source": "relational",
        "hadm_id": 12345,
        "itemid": 50912,
        "valuenum": 1.2,
    },
    {
        "source": "relational",
        "hadm_id": 12345,
        "itemid": 50912,
        "valuenum": 2.5,
    },
    {
        "source": "relational",
        "hadm_id": 12345,
        "itemid": 50912,
        "valuenum": 1.8,
    },
]

answerer = LlamaAnswerer()

answer, prompt_tokens, latency_ms = answerer.answer(
    question, evidence
)

print("Answer:", answer)
print("Expected answer:", 2.5)
print("Prompt tokens:", prompt_tokens)
print("Latency (ms):", latency_ms)
print("Correct:", answer == 2.5)

from clinrag.sources.faiss_notes import FaissNotesSource

notes = FaissNotesSource("data/faiss_index")

result = notes.search(
    "Patient with diabetes receiving insulin",
    k=3
)

for item in result.items:
    print("\nAdmission:", item["hadm_id"])
    print("Score:", item["score"])
    print("Contains insulin:", "insulin" in item["text"].lower())
    print("Full text:", item["text"])



#just for testing purposes
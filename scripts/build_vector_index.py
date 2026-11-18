
from clinrag.sources.faiss_notes import FaissNotesSource

def main():
    notes = FaissNotesSource.build(
        csv_path="data/generated/discharge.csv.gz",
        index_dir="data/faiss_index",
        model_name="BAAI/bge-base-en-v1.5",
        chunk_words=180,
        overlap_words=30,
        batch_size=32,
    )

    print("Total indexed chunks:", notes.total_notes())

    result = notes.search(
        query="Patient with hypertension",
        k=5,
    )

    for item in result.items:
        print("Admission:", item["hadm_id"])
        print("Score:", item["score"])
        print("Text:", item["text"])
        print("-" * 40)


if __name__ == "__main__":
    main()

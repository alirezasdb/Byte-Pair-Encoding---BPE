
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import json


# -----------------------------
# Configuration
# -----------------------------

NUM_STORIES = 5000

OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "corpus_en.jsonl"

DATASET_NAME = "roneneldan/TinyStories"


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading TinyStories dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True
    )

    stories = []

    print(f"Collecting {NUM_STORIES} stories...")

    for item in tqdm(dataset, total=NUM_STORIES):

        text = item.get("text", "")

        # Normalize whitespace
        text = " ".join(text.split())

        if not text:
            continue

        stories.append(text)

        if len(stories) >= NUM_STORIES:
            break

    print("Saving corpus...")

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        for story in stories:

            file.write(
                json.dumps(
                    {"text": story},
                    ensure_ascii=False
                ) + "\n"
            )

    total_chars = sum(len(story) for story in stories)

    print("\nCorpus preparation completed!")

    print(f"Stories: {len(stories)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Average characters: {total_chars / len(stories):.2f}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

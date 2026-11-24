
import json
import os
import re
from tqdm import tqdm
from datasets import load_dataset

# تنظیمات
DATASET_NAME = "codersan/Persian-Wikipedia-Corpus"
NUM_ARTICLES = 5000
MIN_CHARACTERS = 100

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "corpus_fa.jsonl")


def clean_text(text):
    """پاک‌سازی اولیه متن بدون حذف نیم‌فاصله فارسی."""

    if not isinstance(text, str):
        return ""

    # یکسان‌سازی فاصله‌ها و خطوط خالی
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading Persian Wikipedia dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True
    )

    saved_articles = 0
    total_characters = 0

    print("Preparing Persian corpus...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        progress = tqdm(total=NUM_ARTICLES)

        for article in dataset:

            text = clean_text(article.get("Text", ""))

            if len(text) < MIN_CHARACTERS:
                continue

            record = {
                "text": text
            }

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            saved_articles += 1
            total_characters += len(text)

            progress.update(1)

            if saved_articles >= NUM_ARTICLES:
                break

        progress.close()

    print("\nPersian corpus preparation completed!")
    print(f"Articles: {saved_articles}")
    print(f"Total characters: {total_characters:,}")

    if saved_articles:
        print(
            f"Average characters: "
            f"{total_characters / saved_articles:.2f}"
        )

    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

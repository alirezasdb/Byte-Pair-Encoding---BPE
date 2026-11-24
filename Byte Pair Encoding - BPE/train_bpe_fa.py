
import json
import os
import time

from bpe import BPETokenizer
from tqdm import tqdm

# تنظیمات
CORPUS_FILE = "data/corpus_fa.jsonl"
OUTPUT_FILE = "results/vocab_fa.json"

NUM_MERGES = 200
MIN_FREQUENCY = 2


def load_corpus(path):

    corpus = []

    print("Loading Persian corpus...")

    with open(path, "r", encoding="utf-8") as file:

        for line in tqdm(file):
            record = json.loads(line)
            text = record.get("text", "")

            if text.strip():
                corpus.append(text)

    return corpus


def main():

    os.makedirs("results", exist_ok=True)

    corpus = load_corpus(CORPUS_FILE)

    print(f"\nNumber of articles: {len(corpus)}")

    tokenizer = BPETokenizer(
        num_merges=NUM_MERGES,
        min_frequency=MIN_FREQUENCY
    )

    print("\nTraining Persian BPE...")

    start_time = time.time()

    tokenizer.train(corpus)

    training_time = time.time() - start_time

    vocab = tokenizer.get_vocab()

    result = {
        "language": "Persian",
        "algorithm": "Character-level BPE",
        "num_merges_requested": NUM_MERGES,
        "num_merges_completed": len(tokenizer.merges),
        "min_frequency": MIN_FREQUENCY,
        "training_time_seconds": round(training_time, 2),
        "vocab_size": len(vocab),
        "vocab": vocab,
        "token_to_id": tokenizer.token_to_id,
        "merges": tokenizer.merges
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\nPersian BPE Training Results")
    print("-" * 40)

    print(f"Training time: {training_time:.2f} seconds")
    print(f"Requested merges: {NUM_MERGES}")
    print(f"Completed merges: {len(tokenizer.merges)}")
    print(f"Vocabulary size: {len(vocab)}")

    examples = [
        "سلام، حال شما چطور است؟",
        "پردازش زبان طبیعی یکی از شاخه‌های هوش مصنوعی است.",
        "یادگیری ماشین به کامپیوترها امکان یادگیری از داده‌ها را می‌دهد."
    ]

    print("\nTokenization Examples:")

    for text in examples:

        tokens = tokenizer.encode(text)

        print("\nText:", text)
        print("Tokens:", tokens)
        print("Number of tokens:", len(tokens))

        assert tokenizer.decode(tokens) == text

    print("\nTraining completed successfully!")
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()

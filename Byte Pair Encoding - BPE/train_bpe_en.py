
import json
import time

from pathlib import Path

from bpe import BPETokenizer


# =====================================
# Configuration
# =====================================

CORPUS_PATH = Path("data/corpus_en.jsonl")
OUTPUT_PATH = Path("results/vocab_en.json")

NUM_MERGES = 200
MIN_FREQUENCY = 2


def load_corpus(path):

    corpus = []

    print("Loading English corpus...")

    with open(path, "r", encoding="utf-8") as file:

        for line in file:

            item = json.loads(line)

            text = item.get("text", "").strip()

            if text:
                corpus.append(text)

    print(f"Loaded {len(corpus):,} stories.")

    return corpus


def save_vocabulary(tokenizer, output_path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result = {
        "language": "English",
        "algorithm": "Character-level BPE",
        "num_merges_requested": tokenizer.num_merges,
        "num_merges_completed": len(tokenizer.merges),
        "vocab_size": len(tokenizer.vocab),
        "min_frequency": tokenizer.min_frequency,
        "vocab": tokenizer.vocab,
        "token_to_id": tokenizer.token_to_id,
        "merges": [
            list(pair)
            for pair in tokenizer.merges
        ]
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(f"Vocabulary saved to: {output_path}")


def main():

    print("=" * 50)
    print("English BPE Training")
    print("=" * 50)

    # Load data
    corpus = load_corpus(CORPUS_PATH)

    if not corpus:
        raise ValueError("Corpus is empty!")

    # Initialize tokenizer
    tokenizer = BPETokenizer(
        num_merges=NUM_MERGES,
        min_frequency=MIN_FREQUENCY
    )

    # Training
    print("\nTraining BPE...")

    start_time = time.time()

    tokenizer.train(corpus)

    training_time = time.time() - start_time

    # Save vocabulary
    save_vocabulary(tokenizer, OUTPUT_PATH)

    # Results
    print("\n" + "=" * 50)
    print("Training Results")
    print("=" * 50)

    print(f"Training time: {training_time:.2f} seconds")

    print(f"Requested merges: {NUM_MERGES}")

    print(f"Completed merges: {len(tokenizer.merges)}")

    print(f"Vocabulary size: {len(tokenizer.vocab)}")

    # Test sentences
    test_sentences = [
        "Once upon a time there was a little cat.",
        "The cat liked to play with a ball.",
        "Machine learning is an interesting subject."
    ]

    print("\nTokenization Examples")
    print("-" * 50)

    for sentence in test_sentences:

        tokens = tokenizer.encode(sentence)

        print(f"\nText: {sentence}")

        print(f"Tokens: {tokens}")

        print(f"Number of tokens: {len(tokens)}")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()

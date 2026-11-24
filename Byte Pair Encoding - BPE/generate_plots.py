
import csv
import json
import os
import re

import matplotlib.pyplot as plt


# ---------------------------------
# تنظیم مسیرها
# ---------------------------------

EN_CORPUS = "data/corpus_en.jsonl"
FA_CORPUS = "data/corpus_fa.jsonl"

EN_VOCAB = "results/vocab_en.json"
FA_VOCAB = "results/vocab_fa.json"

COMPARISON_FILE = "results/comparison_table.csv"

PLOT_DIR = "results/plots"

os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------------------------
# محاسبه واژگان اولیه
# ---------------------------------

def get_initial_vocabulary(corpus_file):

    vocabulary = set()

    print(f"Reading: {corpus_file}")

    with open(corpus_file, "r", encoding="utf-8") as file:

        for line in file:

            record = json.loads(line)
            text = record.get("text", "")

            # کاراکترهای کلمات
            words = re.findall(r"\S+", text)

            for word in words:
                vocabulary.update(word)

            # توالی‌های فاصله و whitespace
            whitespace = re.findall(r"\s+", text)
            vocabulary.update(whitespace)

    return vocabulary


# ---------------------------------
# محاسبه رشد واژگان
# ---------------------------------



def calculate_vocab_growth(corpus_file, model_file):

    initial_vocab = get_initial_vocabulary(corpus_file)

    with open(model_file, "r", encoding="utf-8") as file:
        model = json.load(file)

    merges = model["merges"]

    vocabulary = set(initial_vocab)
    growth = [len(vocabulary)]

    for first, second in merges:
        vocabulary.add(first + second)
        growth.append(len(vocabulary))

    print(f"\nModel: {model['language']}")
    print(f"Initial vocabulary: {len(initial_vocab)}")
    print(f"Cumulative vocabulary after merges: {len(vocabulary)}")
    print(f"Final stored vocabulary: {model['vocab_size']}")

    return growth


# ---------------------------------
# نمودار رشد واژگان
# ---------------------------------

def plot_vocab_growth():

    print("\nCalculating vocabulary growth...")

    en_growth = calculate_vocab_growth(
        EN_CORPUS,
        EN_VOCAB
    )

    fa_growth = calculate_vocab_growth(
        FA_CORPUS,
        FA_VOCAB
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        range(len(en_growth)),
        en_growth,
        label="English BPE",
        marker="o",
        markersize=2
    )

    plt.plot(
        range(len(fa_growth)),
        fa_growth,
        label="Persian BPE",
        marker="o",
        markersize=2
    )

    plt.title("BPE Vocabulary Growth")
    plt.xlabel("Number of Merge Operations")
    plt.ylabel("Cumulative Vocabulary Size")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output = os.path.join(
        PLOT_DIR,
        "vocab_growth.png"
    )

    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved: {output}")


# ---------------------------------
# نمودار توزیع تعداد توکن‌ها
# ---------------------------------

def plot_token_distribution():

    print("\nReading comparison results...")

    rows = []

    with open(
        COMPARISON_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    labels = [
        f"{row['language'][:2]}-{i + 1}"
        for i, row in enumerate(rows)
    ]

    custom_counts = [
        int(row["custom_bpe_tokens"])
        for row in rows
    ]

    gpt_counts = [
        int(row["gpt4_tokens"])
        for row in rows
    ]

    x = list(range(len(labels)))
    width = 0.35

    plt.figure(figsize=(11, 6))

    custom_positions = [
        value - width / 2 for value in x
    ]

    gpt_positions = [
        value + width / 2 for value in x
    ]

    plt.bar(
        custom_positions,
        custom_counts,
        width,
        label="Custom BPE"
    )

    plt.bar(
        gpt_positions,
        gpt_counts,
        width,
        label="GPT-4 Tokenizer"
    )

    plt.title("Token Distribution Comparison")
    plt.xlabel("Test Samples")
    plt.ylabel("Number of Tokens")

    plt.xticks(x, labels)

    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output = os.path.join(
        PLOT_DIR,
        "token_distribution.png"
    )

    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved: {output}")


# ---------------------------------
# اجرای اصلی
# ---------------------------------

def main():

    print("=" * 55)
    print("Generating BPE Evaluation Plots")
    print("=" * 55)

    plot_vocab_growth()
    plot_token_distribution()

    print("\nAll plots generated successfully!")

    print("\nOutput directory:")
    print(PLOT_DIR)


if __name__ == "__main__":
    main()

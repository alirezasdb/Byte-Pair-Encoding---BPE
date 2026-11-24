
import csv
import json
import os
import time

import tiktoken

from bpe import BPETokenizer


# ---------------------------------
# تنظیمات
# ---------------------------------

EN_MODEL_FILE = "results/vocab_en.json"
FA_MODEL_FILE = "results/vocab_fa.json"

OUTPUT_FILE = "results/comparison_table.csv"

REPEAT_COUNT = 500

# توکنایزر GPT-4
gpt4_tokenizer = tiktoken.get_encoding("cl100k_base")


# ---------------------------------
# بارگذاری مدل BPE آموزش‌دیده
# ---------------------------------

def load_bpe_model(path):

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    tokenizer = BPETokenizer()

    tokenizer.merges = [
        tuple(pair) for pair in data["merges"]
    ]

    tokenizer.is_trained = True

    return tokenizer


# ---------------------------------
# نمونه متن‌ها
# ---------------------------------

samples = [
    {
        "language": "English",
        "text": "Once upon a time there was a little cat."
    },
    {
        "language": "English",
        "text": "The cat liked to play with a ball."
    },
    {
        "language": "English",
        "text": "Machine learning is an interesting subject."
    },
    {
        "language": "Persian",
        "text": "سلام، حال شما چطور است؟"
    },
    {
        "language": "Persian",
        "text": "پردازش زبان طبیعی یکی از شاخه‌های هوش مصنوعی است."
    },
    {
        "language": "Persian",
        "text": "یادگیری ماشین به کامپیوترها امکان یادگیری از داده‌ها را می‌دهد."
    }
]


# ---------------------------------
# اندازه‌گیری زمان اجرا
# ---------------------------------

def measure_time(function, text, repetitions):

    # اجرای اولیه برای آماده‌سازی
    function(text)

    start = time.perf_counter()

    for _ in range(repetitions):
        function(text)

    elapsed = time.perf_counter() - start

    return (elapsed / repetitions) * 1000


# ---------------------------------
# اجرای مقایسه
# ---------------------------------

def main():

    os.makedirs("results", exist_ok=True)

    print("Loading trained BPE models...")

    en_bpe = load_bpe_model(EN_MODEL_FILE)
    fa_bpe = load_bpe_model(FA_MODEL_FILE)

    models = {
        "English": en_bpe,
        "Persian": fa_bpe
    }

    results = []

    print("\nStarting tokenizer comparison...")
    print("=" * 75)

    for sample in samples:

        language = sample["language"]
        text = sample["text"]

        custom_tokenizer = models[language]

        # توکن‌سازی با BPE اختصاصی
        custom_tokens = custom_tokenizer.encode(text)

        # توکن‌سازی با GPT-4
        gpt4_tokens = gpt4_tokenizer.encode(text)

        # بررسی بازسازی متن
        reconstructed = custom_tokenizer.decode(custom_tokens)

        if reconstructed != text:
            raise ValueError(
                f"Text reconstruction failed: {text}"
            )

        # اندازه‌گیری زمان اجرا
        custom_time = measure_time(
            custom_tokenizer.encode,
            text,
            REPEAT_COUNT
        )

        gpt4_time = measure_time(
            gpt4_tokenizer.encode,
            text,
            REPEAT_COUNT
        )

        custom_count = len(custom_tokens)
        gpt4_count = len(gpt4_tokens)

        difference = custom_count - gpt4_count

        percentage = (
            difference / gpt4_count * 100
            if gpt4_count else 0
        )

        result = {
            "language": language,
            "text": text,
            "custom_bpe_tokens": custom_count,
            "gpt4_tokens": gpt4_count,
            "token_difference": difference,
            "difference_percent": round(percentage, 2),
            "custom_bpe_time_ms": round(custom_time, 6),
            "gpt4_time_ms": round(gpt4_time, 6)
        }

        results.append(result)

        print(f"\nLanguage: {language}")
        print(f"Text: {text}")
        print(f"Our BPE tokens: {custom_count}")
        print(f"GPT-4 tokens: {gpt4_count}")
        print(f"Difference: {difference}")
        print(f"Difference percentage: {percentage:.2f}%")
        print(f"Our BPE time: {custom_time:.6f} ms")
        print(f"GPT-4 time: {gpt4_time:.6f} ms")

    # ---------------------------------
    # ذخیره نتایج در CSV
    # ---------------------------------

    fieldnames = list(results[0].keys())

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 75)
    print("Comparison completed successfully!")

    print(f"Number of samples: {len(results)}")
    print(f"Repetitions per sample: {REPEAT_COUNT}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

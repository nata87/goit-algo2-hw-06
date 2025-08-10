import argparse
import re
import sys
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Tuple

import requests
import matplotlib.pyplot as plt


def get_text(source: str, timeout: int = 25) -> str:
    if source.lower().startswith(("http://", "https://")):
        try:
            resp = requests.get(source, timeout=timeout, headers={"User-Agent": "MapReduceWordCount/1.0"})
            resp.raise_for_status()
            resp.encoding = resp.encoding or "utf-8"
            return resp.text
        except requests.RequestException as e:
            print(f"Помилка завантаження: {e}", file=sys.stderr)
            return ""
    else:
        p = Path(source)
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"Помилка читання файлу {p}: {e}", file=sys.stderr)
        else:
            print(f"Файл {p} не знайдено.", file=sys.stderr)
        return ""


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[\w']+\b", text.lower())


def map_function(word: str) -> Tuple[str, int]:
    return word, 1

def shuffle_function(mapped_values: Iterable[Tuple[str, int]]):
    buckets: dict[str, list[int]] = defaultdict(list)
    for key, value in mapped_values:
        buckets[key].append(value)
    return buckets.items()

def reduce_function(key_values: Tuple[str, list[int]]) -> Tuple[str, int]:
    key, values = key_values
    return key, sum(values)

def map_reduce(text: str, workers: int = 8) -> dict[str, int]:
    words = tokenize(text)
    if not words:
        return {}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        mapped_values = list(ex.map(map_function, words))

    shuffled = shuffle_function(mapped_values)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        reduced_pairs = list(ex.map(reduce_function, shuffled))

    return dict(reduced_pairs)


def print_top_words(word_counts: dict[str, int], top_n: int = 10) -> None:
    for i, (w, c) in enumerate(Counter(word_counts).most_common(top_n), 1):
        print(f"{i:>2}. {w}: {c}")

def visualize_top_words(word_counts: dict[str, int], top_n: int = 10) -> None:
    if not word_counts:
        print("Немає даних для візуалізації.")
        return
    top = Counter(word_counts).most_common(top_n)
    labels, values = zip(*top)

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)       
    plt.title(f"Top {top_n} words")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MapReduce — підрахунок частоти слів із багатопотоковістю."
    )
    parser.add_argument(
        "--source",
        default="https://gutenberg.net.au/ebooks01/0100021.txt"
    )

    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    text = get_text(args.source)
    if not text:
        return 1

    counts = map_reduce(text, workers=args.workers)
    print_top_words(counts, top_n=args.top)

    visualize_top_words(counts, top_n=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

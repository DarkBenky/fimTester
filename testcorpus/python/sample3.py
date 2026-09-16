import re
from collections import Counter

WORD = re.compile(r"[a-z']+")


def tokenize(text):
    return WORD.findall(text.lower())


def word_stats(text):
    words = tokenize(text)
    counts = Counter(words)
    longest = max(words, key=len) if words else ""
    return counts, longest


def top_words(counts, limit=3):
    return [word for word, _ in counts.most_common(limit)]


def merge_counts(first, second):
    merged = Counter(first)
    merged.update(second)
    return merged


def main():
    sample = "the quick brown fox jumps over the lazy dog. The fox sleeps."
    counts, longest = word_stats(sample)
    extra = Counter(tokenize("the dog barks and the fox runs"))
    total = merge_counts(counts, extra)
    print("unique:", len(counts))
    print("longest:", longest)
    print("top:", top_words(counts))
    print("merged top:", top_words(total, 5))


if __name__ == "__main__":
    main()

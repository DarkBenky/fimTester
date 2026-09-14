import os
import sys


def process_data(items, threshold):
    results = []
    total = 0
    for item in items:
        value = item.get("value", 0)
        if value > threshold:
            results.append(item)
            total += value
    return results, total


def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as handle:
        return handle.read()


def main(argv):
    name = argv[1] if len(argv) > 1 else "world"
    greeting = f"hello {name}"
    print(greeting)
    data = process_data([{"value": i} for i in range(10)], 4)
    print(data)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

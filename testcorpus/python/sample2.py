import json
from dataclasses import dataclass


@dataclass
class Item:
    name: str
    price: float
    quantity: int = 0

    def total(self):
        return round(self.price * self.quantity, 2)


class Inventory:
    def __init__(self):
        self.items = {}

    def add(self, item):
        self.items[item.name] = item
        return item

    def remove(self, name):
        if name not in self.items:
            raise KeyError(name)
        return self.items.pop(name)

    def value(self):
        return round(sum(item.total() for item in self.items.values()), 2)

    def low_stock(self, threshold):
        return sorted(name for name, item in self.items.items() if item.quantity < threshold)


def load_inventory(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    inventory = Inventory()
    for entry in data:
        inventory.add(Item(**entry))
    return inventory


def main():
    inventory = Inventory()
    inventory.add(Item("bolt", 0.25, 120))
    inventory.add(Item("nut", 0.10, 8))
    inventory.add(Item("washer", 0.05, 340))
    print("value:", inventory.value())
    print("low:", inventory.low_stock(50))


if __name__ == "__main__":
    main()

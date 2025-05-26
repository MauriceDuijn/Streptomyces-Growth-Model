from src.utils.instance_tracker import InstanceTracker


class Element(InstanceTracker):
    def __init__(self, name: str, symbol: str = None, initial_amount: int = 0) -> None:
        super().__init__()
        self.name: str = name
        self.symbol: str = symbol
        self.amount: int = initial_amount

    def __repr__(self) -> str:
        return f"Element({self.name}, X({self.amount}))"


if __name__ == '__main__':
    hydrogen = Element("Hydrogen", 'H', 2)
    oxygen = Element("Oxygen", 'O', 1)
    print(hydrogen, oxygen)

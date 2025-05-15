class Element:
    total_elements: int = 0
    elemental_species: list['Element'] = []

    def __init__(self, name: str, symbol: str = None, initial_amount: int = 0) -> None:
        self.name: str = name
        self.symbol: str = symbol
        self.amount: int = initial_amount
        self.index: int = self.total_elements
        Element.total_elements += 1
        self.elemental_species.append(self)

    def __repr__(self) -> str:
        return f"Element({self.name}, X({self.amount}))"


if __name__ == '__main__':
    hydrogen = Element("Hydrogen", 'H', 2)
    oxygen = Element("Oxygen", 'O', 1)
    print(hydrogen, oxygen)

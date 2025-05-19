from math import factorial
import numpy as np
from Chemistry_manager.ElementalSpecies import Element


class Reaction:
    total_reactions = 0
    reaction_channels: list['Reaction'] = []

    def __init__(self, name: str, rate: float,
                 reactants: dict[Element, int],
                 products: dict[Element, int]):
        self.name: str = name
        self.rate: np.float64 = np.float64(rate)
        self.reactants: dict[Element, int] = reactants
        self.products: dict[Element, int] = products
        self.propensity: np.float64 = np.float64(0)
        self.index: int = self.total_reactions
        Reaction.total_reactions += 1
        self.reaction_channels.append(self)

    def __str__(self) -> str:
        return f"Reaction({self.name}, prop.:{self})"

    def react(self) -> None:
        # Reduce the amount of reactant species
        for element, coefficient in self.reactants.items():
            element.amount -= coefficient

        # Add the amount of created product species
        for element, coefficient in self.products.items():
            element.amount += coefficient

    def calc_reactorial_count(self) -> float:
        """Calculates all distinct combinations of the reactants."""
        h = 1.0
        for element, coefficient in self.reactants.items():
            if coefficient == 1:
                h *= element.amount
            elif coefficient == 2:
                h *= element.amount * (element.amount - 1) / 2
            else:
                factors = np.arange(element.amount, element.amount - coefficient, -1, dtype=np.float64)
                h *= np.prod(factors) / factorial(coefficient)
        return h

    def calc_propensity(self) -> None:
        self.propensity = self.rate * self.calc_reactorial_count()

    @classmethod
    def reset_class(cls):
        cls.total_reactions = 0
        cls.reaction_channels = []

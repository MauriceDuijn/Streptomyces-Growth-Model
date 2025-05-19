import numpy as np
from utils.DynamicArray import Dynamic2DArray
from Cell_manager.Cell import Cell


class Condition:
    total_conditions = 0
    condition_collection: list['Condition'] = []
    cell_condition_factor_array: Dynamic2DArray = Dynamic2DArray()

    @classmethod
    def add_new_condition(cls, instance: 'Condition'):
        instance.index = cls.total_conditions
        cls.total_conditions += 1
        cls.condition_collection.append(instance)
        cls.cell_condition_factor_array.add_column()

    def __init__(self, name: str, method_name: str, parameter: str, alpha: float = 1, threshold=1):
        self.name = name
        self.method_func = getattr(self, method_name)
        self.param_arr = getattr(Cell, f"{parameter}_array")
        self.alpha = alpha
        self.threshold = threshold

        self.index = None
        self.add_new_condition(self)

    @property
    def factor(self) -> np.ndarray:
        return self.cell_condition_factor_array[:, self.index]

    @factor.setter
    def factor(self, value):
        self.cell_condition_factor_array[:, self.index] = value

    @classmethod
    def combined_factor(cls, condition_indexes):
        # print("condition values", cls.cell_condition_factor_array[:, condition_indexes])
        # print(type(cls.cell_condition_factor_array[:, condition_indexes]))
        # print("condition prod", np.prod(cls.cell_condition_factor_array[:, condition_indexes], axis=1))
        return np.prod(cls.cell_condition_factor_array[:, condition_indexes], axis=1)

    def threshold_reduction(self):
        np.maximum(self.param_arr.active - self.threshold, 0,
                   out=self.factor)

    def calc_factor(self):
        # Quick return if condition is static
        if self.method_func == self.static:
            return

        # Quick execution if condition is constant
        if self.method_func == self.constant:
            self.method_func()
            return

        # Reduces the current amount by the set threshold, minimum of 0 (non-negative values)
        self.threshold_reduction()

        # Update the factor based on the given parameter and method
        self.method_func()

    def static(self):
        """
        The factor is static and is only changed when a specific action occurs.

        These actions are:
            - CalcInfluence
        """
        return self.factor

    def constant(self):
        self.factor = self.alpha

    def linear(self):
        self.factor *= self.alpha

    def powerlaw(self):
        self.factor **= self.alpha

    def exponential(self):
        np.exp(self.alpha * self.cell_condition_factor_array[:, self.index],
               out=self.factor)

    @classmethod
    def reset_class(cls):
        cls.total_conditions = 0
        cls.condition_collection = []
        cls.cell_condition_factor_array = Dynamic2DArray()


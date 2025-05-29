import numpy as np
from numba import njit
from src.utils.dynamic_array import DynamicArray, Dynamic2DArray
from src.utils.instance_tracker import InstanceTracker
from src.algorithm.cell_based.cell import Cell


class Condition(InstanceTracker):
    cell_condition_factor_array: Dynamic2DArray = Dynamic2DArray()
    METHOD_MODES = {
        "constant": 0,
        "linear": 1,
        "powerlaw": 2,
        "exponential": 3,
    }

    def __init__(self, name: str, method_name: str, parameter: str, alpha: float = 1, threshold=1):
        super().__init__()
        self.cell_condition_factor_array.add_column()

        self.name = name
        self.method_name: str = method_name
        self.param_arr: DynamicArray or Dynamic2DArray = getattr(Cell, f"{parameter}_array")
        self.alpha: float = alpha
        self.threshold: float = threshold

    @property
    def factor(self) -> np.ndarray:
        return self.cell_condition_factor_array[:, self.index]

    @factor.setter
    def factor(self, value):
        self.cell_condition_factor_array[:, self.index] = value

    # def threshold_reduction(self):
    #     np.maximum(self.param_arr.active - self.threshold, 0,
    #                out=self.factor)

    def calc_factor(self):
        # Quick return if condition is static
        if self.method_name == "static":
            return

        mode = self.METHOD_MODES[self.method_name]
        self.update_condition_factors_turbo(
            self.cell_condition_factor_array[:, self.index],
            self.param_arr.active,
            self.alpha,
            self.threshold,
            mode
        )

    @staticmethod
    @njit
    def update_condition_factors_turbo(factor_column, param_column, alpha, threshold, mode: int):
        """
        mode: 0 = constant, 1 = linear, 2 = powerlaw, 3 = exponential
        """
        for i in range(factor_column.shape[0]):
            val = max(param_column[i] - threshold, 0)
            if mode == 0:
                factor_column[i] = alpha
            elif mode == 1:
                factor_column[i] = val * alpha
            elif mode == 2:
                factor_column[i] = val ** alpha
            elif mode == 3:
                factor_column[i] = np.exp(val * alpha)

    @classmethod
    def reset_class(cls):
        super().reset_class()
        cls.cell_condition_factor_array = Dynamic2DArray()


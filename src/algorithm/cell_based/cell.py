import numpy as np
from src.utils.instance_tracker import InstanceTracker
from src.utils.dynamic_array import DynamicArray, Dynamic2DArray
from src.algorithm.event.state import State


class Cell(InstanceTracker):
    center_point_array: Dynamic2DArray = Dynamic2DArray(capacity_columns=2)
    end_point_array: Dynamic2DArray = Dynamic2DArray(capacity_columns=2)
    age_array: DynamicArray = DynamicArray()
    crowding_index_array: DynamicArray = DynamicArray()
    DivIVA_array: DynamicArray = DynamicArray()
    DivIVA_binding_rate: float = 0

    def __init__(self, center_position, end_position,
                 direction, length=1,
                 parent=None, state=None, colony_index=None):
        super().__init__()
        self.state: State = state                           # Initial cell state, changes dynamic
        self.parent: Cell = parent                          # Parent.end is start position
        self.children: list[Cell] = []                      # List of all the daughter cells (children)
        self.center_point_array.append(center_position)     # Center point of the cell, used for distance calculations
        self.end_point_array.append(end_position)           # End position, new cells grow relative from this point
        self.direction: float = np.radians(direction)       # Direction the cell is facing (converts degrees to radians)
        self.length: float = length                         # The length of the cell, from start position to end position

        self.colony_index = colony_index
        self.extend_matrices()

    @property
    def age(self):
        return self.age_array[self.index]

    @property
    def center(self):
        return self.center_point_array[self.index]

    @property
    def end(self):
        return self.end_point_array[self.index]

    @property
    def crowding(self) -> float:
        return self.crowding_index_array[self.index]

    @crowding.setter
    def crowding(self, value):
        self.crowding_index_array[self.index] = value

    @property
    def DivIVA(self):
        return self.DivIVA_array[self.index]

    @DivIVA.setter
    def DivIVA(self, value):
        self.DivIVA_array[self.index] = value

    def extend_matrices(self):
        self.age_array.append(0)
        self.crowding_index_array.append(0)
        self.DivIVA_array.append(0)

    def link_child(self, child: 'Cell'):
        self.children.append(child)

    @staticmethod
    def create_root_cell(position: tuple[float, float], direction: float, start_state: State) -> 'Cell':
        return Cell(position, position, direction, state=start_state)

    @classmethod
    def increase_age_over_time(cls, time_step):
        cls.age_array.active += time_step

    @classmethod
    def increase_polarisome_over_time(cls, time_step):
        cls.DivIVA_array.active *= np.exp(cls.DivIVA_binding_rate * time_step)

    @classmethod
    def reset_class(cls):
        super().reset_class()

        cls.center_point_array: Dynamic2DArray = Dynamic2DArray(capacity_columns=2)
        cls.end_point_array: Dynamic2DArray = Dynamic2DArray(capacity_columns=2)

        cls.age_array: DynamicArray = DynamicArray()
        cls.crowding_index_array: DynamicArray = DynamicArray()
        cls.DivIVA_array: DynamicArray = DynamicArray()
        cls.DivIVA_binding_rate: float = 0

    @classmethod
    def load_data(cls, cell_data: dict[str, int or float or list[int]]):
        """Assuming the cell array data is loaded in first."""
        cell_index = cell_data["index"]
        parent: int = cell_data["parent"]
        if parent is not None:
            parent: Cell = cls.instances[parent]
        state = State.instances[cell_data["state_index"]]

        new_cell = cls(
            cls.center_point_array[cell_index],
            cls.end_point_array[cell_index],
            cell_data["direction"],
            cell_data["length"],
            parent=parent,
            state=state,
            colony_index=cell_data["colony_index"]
        )

        if parent is not None:
            parent.children.append(new_cell)

        return new_cell



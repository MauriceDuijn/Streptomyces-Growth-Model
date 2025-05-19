import numpy as np
from utils.DynamicArray import DynamicArray, Dynamic2DArray
from Event_manager.State import State


class Cell:
    total_cells = 0
    cell_collection: list['Cell'] = []

    center_point_array: Dynamic2DArray = Dynamic2DArray(capacity_columns=2)
    end_point_array: Dynamic2DArray = Dynamic2DArray(capacity_columns=2)

    age_array: DynamicArray = DynamicArray()
    crowding_index_array: DynamicArray = DynamicArray()
    DivIVA_array: DynamicArray = DynamicArray()
    DivIVA_binding_rate: float = 0

    @classmethod
    def add_new_cell(cls, instance: 'Cell'):
        instance.index = cls.total_cells
        cls.total_cells += 1

        # Initial dummy data
        cls.age_array.append(0)
        cls.crowding_index_array.append(0)
        cls.DivIVA_array.append(0)
        cls.cell_collection.append(instance)  # Store the cell in collective list

    def __init__(self, center_position, end_position,
                 direction, length=1,
                 parent=None, state=None):
        self.state: State = state                           # Initial cell state, changes dynamic
        self.parent: Cell = parent                          # Parent.end is start position
        self.children: list[Cell] = []                      # List of all the daughter cells (children)
        self.center_point_array.append(center_position)     # Center point of the cell, used for distance calculations
        self.end_point_array.append(end_position)           # End position, new cells grow relative from this point
        self.direction: float = np.radians(direction)       # Direction the cell is facing (converts degrees to radians)
        self.length: float = length                         # The length of the cell, from start position to end position

        self.index = None
        self.add_new_cell(self)
        self.colony_index = None

    @staticmethod
    def create_root_cell(position: tuple[float, float], direction: float, start_state: State) -> 'Cell':
        return Cell(position, position, direction, state=start_state)

    def link_child(self, child: 'Cell'):
        self.children.append(child)

    @classmethod
    def increase_age_over_time(cls, time_step):
        cls.age_array.active += time_step

    @classmethod
    def increase_polarisome_over_time(cls, time_step):
        cls.DivIVA_array.active *= np.exp(cls.DivIVA_binding_rate * time_step)

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
    def crowding_index(self) -> float:
        return self.crowding_index_array[self.index]

    @crowding_index.setter
    def crowding_index(self, value):
        self.crowding_index_array[self.index] = value

    @property
    def DivIVA(self):
        return self.DivIVA_array[self.index]

    @DivIVA.setter
    def DivIVA(self, value):
        self.DivIVA_array[self.index] = value

    @classmethod
    def reset_class(cls):
        cls.total_cells = 0
        cls.cell_collection: list['Cell'] = []

        cls.center_point_array: Dynamic2DArray = Dynamic2DArray(capacity_columns=2)
        cls.end_point_array: Dynamic2DArray = Dynamic2DArray(capacity_columns=2)

        cls.age_array: DynamicArray = DynamicArray()
        cls.crowding_index_array: DynamicArray = DynamicArray()
        cls.DivIVA_array: DynamicArray = DynamicArray()
        cls.DivIVA_binding_rate: float = 0


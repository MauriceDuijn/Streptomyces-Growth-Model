import numpy as np
from utils.DynamicArray import DynamicArray
from Space_manager.SpatialHashing import SpatialHashing
from Cell_manager.Cell import Cell
from Event_manager.Condition import Condition


class Colony:
    total_colonies = 0
    colonies: list['Colony'] = []

    @classmethod
    def add_new_colony(cls, instance: 'Colony'):
        instance.index = cls.total_colonies
        instance.root.colony_index = cls.total_colonies
        cls.total_colonies += 1
        cls.colonies.append(instance)

    def __init__(self, root_cell: Cell):
        self.root = root_cell
        self.cell_indexes = DynamicArray(data_type=np.int32)
        self.cell_grid = SpatialHashing()

        self.index = None
        self.add_new_colony(self)

    def add_cell(self, cell: Cell):
        cell.colony_index = self.index
        self.cell_indexes.append(cell.index)
        self.cell_grid.insert(cell)

    def add_branch(self, branch: list[Cell]):
        for cell in branch:
            self.add_cell(cell)

    def remove_branch(self, branch: list[Cell]):
        branch_ids: list[int] = [cell.index for cell in branch]
        self.cell_indexes.batch_remove(branch_ids)

    def remove_crowding_index(self, branch: list[Cell], crowding_condition: Condition):
        for cell in branch:
            self.cell_grid.query(cell.center)

    @classmethod
    def get_cell_indexes(cls, colony_ind: int):
        return cls.colonies[colony_ind].cell_indexes.active

    @classmethod
    def get_all_neighbours(cls, cell: Cell):
        return cls.colonies[cell.colony_index].cell_grid.query(cell.center)

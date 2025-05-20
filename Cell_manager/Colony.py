import numpy as np
from utils.DynamicArray import DynamicArray
from Space_manager.SpatialHashing import SpatialHashing
from Cell_manager.Cell import Cell
from Event_manager.Condition import Condition


class Colony:
    total_colonies = 0
    colonies: list['Colony'] = []

    def __init__(self, root_cell: Cell):
        self.root = root_cell
        self.cell_indexes = DynamicArray(data_type=np.int32)
        self.cell_grid = SpatialHashing()

        self.index = None
        self.add_new_colony(self)

    @property
    def cell_count(self):
        return len(self.cell_indexes)

    @property
    def cell_points(self):
        return Cell.center_point_array[self.cell_indexes]

    @property
    def extent(self):
        points = self.cell_points
        min_x, max_x = min(points[:, 0]), max(points[:, 0])
        min_y, max_y = min(points[:, 1]), max(points[:, 1])
        return min_x, min_y, max_x, max_y

    @property
    def box_diameter(self):
        """box width and height"""
        min_x, min_y, max_x, max_y = self.extent
        return max_x - min_x, max_y - min_y

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
    def add_new_colony(cls, instance: 'Colony'):
        instance.index = cls.total_colonies
        instance.root.colony_index = cls.total_colonies
        cls.total_colonies += 1
        cls.colonies.append(instance)

    @classmethod
    def get_cell_indexes(cls, colony_ind: int):
        return cls.colonies[colony_ind].cell_indexes.active

    @classmethod
    def get_all_neighbours(cls, cell: Cell):
        return cls.colonies[cell.colony_index].cell_grid.query(cell.center)

    @classmethod
    def reset_class(cls):
        cls.total_colonies = 0
        cls.colonies = []

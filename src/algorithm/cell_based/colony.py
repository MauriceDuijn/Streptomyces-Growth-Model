import numpy as np
from src.utils.dynamic_array import DynamicArray
from src.utils.instance_tracker import InstanceTracker
from src.algorithm.spatial.spatial_hashing import SpatialHashing
from src.algorithm.cell_based.cell import Cell


class Colony(InstanceTracker):
    def __init__(self, root_cell: Cell):
        super().__init__()
        self.root = root_cell
        self.cell_indexes = DynamicArray(data_type=np.int32)
        self.cell_grid = SpatialHashing()
        self.add_cell(self.root)

    @property
    def cell_count(self):
        return len(self.cell_indexes)

    @property
    def cell_points(self):
        return Cell.center_point_array[self.cell_indexes]

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

    @classmethod
    def get_cell_indexes(cls, colony_ind: int):
        return cls.instances[colony_ind].cell_indexes.active

    @classmethod
    def get_all_neighbours(cls, cell: Cell):
        return cls.instances[cell.colony_index].cell_grid.query(cell.center)

    @classmethod
    def load_data(cls, colony_data: dict[str, int or list[int]]):
        """Assuming the cell class is fully loaded in first."""
        return cls(
            root_cell=Cell.instances[colony_data["root_index"]]
        )




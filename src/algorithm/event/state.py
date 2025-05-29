import numpy as np
from src.utils.dynamic_array import Dynamic2DArray
from src.utils.instance_tracker import InstanceTracker


class State(InstanceTracker):
    event_mask_array: Dynamic2DArray = Dynamic2DArray()     # States as rows and Events as columns
    cell_mask_array: Dynamic2DArray = Dynamic2DArray()      # Cells as rows and States as columns

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.event_mask_array.append(0)

    @property
    def event_mask(self) -> np.ndarray:
        return self.event_mask_array[self.index]

    def add_event_mask(self, event_indexes):
        self.event_mask_array[self.index, event_indexes] = np.float64(1)

    @classmethod
    def reset_class(cls):
        super().reset_class()
        cls.event_mask_array = Dynamic2DArray()
        cls.cell_mask_array = Dynamic2DArray()


if __name__ == '__main__':
    spore_state = State('Root')
    print(spore_state)

import numpy as np
from src.utils.dynamic_array import Dynamic2DArray


class State:
    total_states: int = 0
    state_nodes: list['State'] = []
    event_mask_array: Dynamic2DArray = Dynamic2DArray()
    cell_mask_array: Dynamic2DArray = Dynamic2DArray()

    def __init__(self, name: str):
        self.name = name
        self.index = self.total_states
        State.total_states += 1
        self.event_mask_array.append(0)
        self.state_nodes.append(self)

    def __str__(self):
        return f"State({self.name})"

    @property
    def event_mask(self) -> np.ndarray:
        return self.event_mask_array[self.index]

    def add_event_mask(self, event_indexes):
        self.event_mask_array[self.index, event_indexes] = np.float64(1)

    @classmethod
    def reset_class(cls):
        cls.total_states = 0
        cls.state_nodes = []
        cls.event_mask_array = Dynamic2DArray()
        cls.cell_mask_array = Dynamic2DArray()


if __name__ == '__main__':
    spore_state = State('Root')
    print(spore_state)

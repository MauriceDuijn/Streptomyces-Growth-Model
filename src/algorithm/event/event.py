import numpy as np
from bisect import bisect_left
from src.utils.dynamic_array import Dynamic2DArray
from src.utils.instance_tracker import InstanceTracker
from src.algorithm.cell_based.cell import Cell
from src.algorithm.cell_based.cell_action import Action, SwitchState
from src.algorithm.chemistry.reaction import Reaction
from src.algorithm.event.state import State
from src.algorithm.event.condition import Condition

from numba import njit


class Event(InstanceTracker):
    event_propensities_array: Dynamic2DArray = Dynamic2DArray()
    conditions_indexes_array: Dynamic2DArray = Dynamic2DArray(capacity_columns=Condition.total)

    def __init__(self, name: str,
                 ingoing_states: list[State],
                 outgoing_state: State,
                 conditions: list[Condition],
                 action: Action,
                 chemical_channel: Reaction):
        super().__init__()
        self.name: str = name
        self.ingoing_states: list[State] = ingoing_states
        self.out_state: SwitchState = SwitchState(outgoing_state)
        self.conditions_indexes: list[int] = [condition.index for condition in conditions]
        self.action: Action = action
        self.reaction: Reaction = chemical_channel

        self.add_state_mask()                       # Link the ingoing states with the current event
        self.event_propensities_array.add_column()  # Add a extra column in the event propensity array

    @property
    def propensity(self) -> np.ndarray:
        return self.event_propensities_array[:, self.index]

    def update_propensity(self):
        # Use reaction propensity as base
        self.event_propensities_array[:, self.index] = self.reaction.propensity

        # Combine the condition factors
        for ind in self.conditions_indexes:
            self.event_propensities_array[:, self.index] *= Condition.cell_condition_factor_array[:, ind]

    def update(self, cell: Cell):
        # Chemical channel update
        self.reaction.react()

        # Physical update
        self.action.update(cell)

        # Update the state of the cell
        self.out_state.update(cell)

    def add_state_mask(self):
        # Add a new column for current event to the event masks
        State.event_mask_array.add_column()
        State.cell_mask_array.add_column()

        # Update the state mask for every ingoing state so it execute the event
        for in_state in self.ingoing_states:
            in_state.add_event_mask(self.index)

    @classmethod
    def get_total_propensity(cls):
        """
        Get the total sum propensity over the entire event propensity matrix.

        :return: total event propensity
        """
        return cls.event_propensities_array.active.sum()

    @classmethod
    def state_mask_correction(cls):
        cls.event_propensities_array.arr *= State.cell_mask_array.arr

    @classmethod
    def update_total_propensity(cls):
        cls.total_propensity = cls.event_propensities_array.active.sum()

    @classmethod
    def random_cell_event_index(cls, total_propensity: float) -> tuple[int, int]:
        """
        Picks a random event based on the propensity of an event of each cell.

        Convert the propensity of each cell-event in the event matrix to a probability by dividing by total propensity.
        The probability is in range 0 to 1, and the sum of all probabilities is equal to 1.
        Flatten the matrix to an array and pick a random index via choice (choice handles the random value internally).
        Break the random flat index up into the cell index and event index via divmod.

        Via the cell index and the event index, the specific cell and event can be traced from their lists.

        :return: A random cell index and event index weighted on their event propensity.
        """
        r = np.random.uniform(0, total_propensity)
        return divmod(cls.find_index(cls.event_propensities_array.active.ravel(), r), cls.total)

    @staticmethod
    @njit
    def find_index(arr, r):
        total = 0.0
        for i in range(arr.size):
            total += arr[i]
            if r < total:
                return i
        return arr.size - 1

    @classmethod
    def print_event_matrix(cls):
        state_max_length = 0
        for state in State.instances:
            state_max_length = max(state_max_length, len(state.name))
        event_max_length = 0
        for event in cls.instances:
            event_max_length = max(event_max_length, len(event.name))

        state_padding = ' ' * state_max_length
        event_name_header = ' '.join(f"{event.name:^{event_max_length}}" for event in Event.instances)

        print(state_padding, event_name_header)
        for cell_ind, line in enumerate(cls.event_propensities_array.active):
            cell_state = f"{Cell.instances[cell_ind].state.name:^{state_max_length}}"
            event_propensities = ' '.join(f"{data:^{event_max_length}}" for data in line)
            print(cell_state, event_propensities)

    @classmethod
    def reset_class(cls):
        super().reset_class()
        cls.event_propensities_array = Dynamic2DArray()
        cls.total_propensity = 0

    @classmethod
    def random_cell_event_index_old(cls) -> tuple[int, int]:
        """
        [Old slower method but skips njit]

        Picks a random event based on the propensity of an event of each cell.

        Convert the propensity of each cell-event in the event matrix to a probability by dividing by total propensity.
        The probability is in range 0 to 1, and the sum of all probabilities is equal to 1.
        Flatten the matrix to an array and pick a random index via choice (choice handles the random value internally).
        Break the random flat index up into the cell index and event index via divmod.

        Via the cell index and the event index, the specific cell and event can be traced from their lists.

        :return: A random cell index and event index based on their event propensity.
        """
        r = np.random.uniform(0, cls.total_propensity)
        probabilities = np.cumsum(cls.event_propensities_array.active)
        flat_random_index = bisect_left(probabilities, r)
        cell_index, event_index = divmod(flat_random_index, cls.total)
        return cell_index, event_index



import numpy as np
from bisect import bisect_left
from utils.DynamicArray import Dynamic2DArray
from Cell_manager.Cell import Cell
from Cell_manager.CellAction import Action, SwitchState
from Chemistry_manager.ReactionChannel import Reaction
from Event_manager.State import State
from Event_manager.Condition import Condition


class Event:
    total_events = 0
    event_instances: list['Event'] = []
    event_propensities_array: Dynamic2DArray = Dynamic2DArray()
    total_propensity: float = 0

    def __init__(self, name: str,
                 ingoing_states: list[State],
                 outgoing_state: State,
                 conditions: list[Condition],
                 action: Action,
                 chemical_channel: Reaction):
        self.name: str = name
        self.conditions_indexes: list[int] = [condition.index for condition in conditions]
        self.ingoing_states: list[State] = ingoing_states
        self.out_state: SwitchState = SwitchState(outgoing_state)
        self.action: Action = action
        self.reaction: Reaction = chemical_channel

        self.index = self.total_events              # Set index
        Event.total_events += 1                     # Increase total events amount
        self.add_state_mask()                       # Link the ingoing states with the current event
        self.event_propensities_array.add_column()  # Add a extra column in the event propensity array
        self.event_instances.append(self)           # Place it in the set

    @property
    def propensity(self) -> np.ndarray:
        return self.event_propensities_array[:, self.index]

    @propensity.setter
    def propensity(self, value):
        self.event_propensities_array[:, self.index] = value

    def update_propensity(self):
        self.propensity = self.reaction.propensity * self.event_factor()

    def event_factor(self):
        return Condition.combined_factor(self.conditions_indexes)

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
    def state_mask_correction(cls):
        cls.event_propensities_array.arr *= State.cell_mask_array.arr

    @classmethod
    def update_total_propensity(cls):
        cls.total_propensity = cls.event_propensities_array.sum()

    @classmethod
    def random_cell_event_index(cls) -> tuple[int, int]:
        """
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
        cell_index, event_index = divmod(flat_random_index, cls.total_events)

        return cell_index, event_index

    @classmethod
    def print_event_matrix(cls):
        state_max_length = 0
        for state in State.state_nodes:
            state_max_length = max(state_max_length, len(state.name))
        event_max_length = 0
        for event in cls.event_instances:
            event_max_length = max(event_max_length, len(event.name))

        state_padding = ' ' * state_max_length
        event_name_header = ' '.join(f"{event.name:^{event_max_length}}" for event in Event.event_instances)

        print(state_padding, event_name_header)
        for cell_ind, line in enumerate(cls.event_propensities_array.active):
            cell_state = f"{Cell.cell_collection[cell_ind].state.name:^{state_max_length}}"
            event_propensities = ' '.join(f"{data:^{event_max_length}}" for data in line)
            print(cell_state, event_propensities)

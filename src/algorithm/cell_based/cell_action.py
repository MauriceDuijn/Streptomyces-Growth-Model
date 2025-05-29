from abc import ABC, abstractmethod
import numpy as np
from numba import njit

from src.utils.benchmark_timer import Timer
from src.utils.dynamic_array import Dynamic2DArray
from src.algorithm.spatial.spatial_hashing import SpatialHashing
from src.algorithm.cell_based.cell import Cell
from src.algorithm.cell_based.colony import Colony
from src.algorithm.event.state import State
from src.algorithm.event.condition import Condition


action_timer = Timer()


class Action(ABC):
    """
    Parent Action class.
    Stores all global variables and gives a parent type to the wide variety of actions.

    Stores:
    Space: Quadtree that holds the points of the cells
    Cell collection: All the cells
    Cell condition: Cell condition matrix
    Event matrix: The matrix of possible events in the matrix
    """
    space: SpatialHashing = SpatialHashing
    cell_collection: list[Cell] = None
    state_mask: Dynamic2DArray = None
    event_propensities: Dynamic2DArray = None
    condition_factors: Dynamic2DArray = None

    @abstractmethod
    def update(self, cell: Cell):
        """
        Each action has a unique update.
        The base class has no specified function and raises a error.

        :param cell: Cell where the changes are applied on.
        """
        raise NotImplementedError("The parent class function is not used. Use one of the subclasses of Action.")

    def add_base(self):
        self.event_propensities.append(0)
        self.state_mask.append(0)
        self.condition_factors.append(0)


class DuoAction:
    """
    Updates an action on two cells.
    """
    def update(self, cell_1: Cell, cell_2: Cell):
        raise NotImplementedError("The parent class function is not used. Use one of the subclasses of Action.")


class SwitchState(Action):
    def __init__(self, new_state):
        self.new_state = new_state

    @action_timer.measure_decorator("SwitchState")
    def update(self, cell: Cell):
        cell.state = self.new_state
        State.cell_mask_array[cell.index] = cell.state.event_mask


class CellGeometryCalculator:
    """Handles geometric calculations for new cell positions"""

    def __init__(self, length, angle_deviation=0, bend=0):
        self.length = length
        self.angle_deviation = np.radians(angle_deviation)
        self.bend = np.radians(bend)

    def calculate_new_cell_points(self, parent_cell: Cell, tropism_bend: float = 0) -> (tuple[int, int], tuple[int, int], float):
        """
        Calculate new cell's center, end, and direction.

        :param parent_cell: Used to copy the direction
        :param tropism_bend: Bend towards nutrients, away from crowded spaces (in radians)
        :return: Cell center point, Cell end point, direction (in radians)
        """
        noise = np.random.normal(0, self.angle_deviation)
        random_bend = self.bend if np.random.random() < 0.5 else -self.bend
        new_direction = parent_cell.direction + random_bend + tropism_bend + noise

        return self.spatial_calculations(parent_cell, new_direction)

    def spatial_calculations(self, cell: Cell, new_direction: float) -> (tuple[int, int], tuple[int, int], float):
        x, y = cell.end
        dx = self.length * np.sin(new_direction)
        dy = self.length * np.cos(new_direction)

        new_center = (x + (dx / 2), y + (dy / 2))
        new_end = (x + dx, y + dy)

        return new_center, new_end, new_direction


class TropismCalculator:
    """Handles tropism-related calculations"""

    def __init__(self, sensitivity=0, max_bend=0):
        self.alpha = sensitivity
        self.max_bend = np.radians(max_bend)
        self.sampler_distance = 1e-6
        self.half_pi = np.pi / 2

    def calc_tropism_bend(self, cell: Cell) -> float:
        """
        Calculate tropism bend based on crowding stimuli.

        At the end point of the cell a left and right sampler point is generated.
        Each sampler calculates the crowding index (stimulus) of neighbouring cells.
        The difference of stimulus between the samplers is converted to a bend (away from the more crowded area).
        A positive bend is counter-clockwise turn and negative bend is clockwise.

        :param cell: Parent cell that calculates the tropism bend relative to the child cell
        :return: Tropism bend (in radians)
        """
        if self.alpha == 0:
            return 0

        neighbour_indexes = Colony.get_all_neighbours(cell)

        left_sample, right_sample = self._get_sampler_points(cell)
        left_stimulus = self._calc_total_stimulus(neighbour_indexes, left_sample)
        right_stimulus = self._calc_total_stimulus(neighbour_indexes, right_sample)

        # left_stimulus, right_stimulus = self._calc_total_stimulus(neighbour_indexes, left_sample, right_sample)
        difference = (right_stimulus - left_stimulus) / self.sampler_distance

        bend = np.tanh(difference * self.alpha) * self.max_bend

        return bend

    def _get_sampler_points(self, cell: Cell) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Create the geometric points of the sampler points.
        At +90 degrees is the left sampler, at -90 degrees is the right sampler,
        both relative to the end point.

        :param cell:
        :return:
        """
        x, y = cell.end
        direction = cell.direction

        dx = np.sin(direction + self.half_pi) * self.sampler_distance
        dy = np.cos(direction + self.half_pi) * self.sampler_distance

        left_sampler_point = (x + dx, y + dy)
        right_sampler_point = (x - dx, y - dy)

        return left_sampler_point, right_sampler_point

    @staticmethod
    def _calc_total_stimulus(neighbour_indexes, sample_points):
        """Calculate the total crowding stimulus from a single sample point"""
        points = Cell.center_point_array[neighbour_indexes]
        dists = Action.space.calc_distances(points, sample_points)
        filtered_dists = Action.space.filter_distances(dists)
        return CrowdingIndex.calc_base_crowding_index(filtered_dists).sum()


class GrowCell(Action, CellGeometryCalculator, TropismCalculator):
    """
    Next update a new cell will grow from given cell.
    """
    cell_length: float = 0
    angle_deviation: float = 0
    tropism_sensitivity: float = 0
    tropism_max_bend: float = 0

    def __init__(self,
                 parent_cell_actions: list[Action],
                 new_cell_actions: list[Action],
                 dou_actions: list[DuoAction],
                 cell_length=None, angle_deviation=None, bend=0,
                 tropism_sensitivity=None, tropism_max_bend=None):
        self.parent_actions = parent_cell_actions
        self.new_actions = new_cell_actions
        self.relation_actions = dou_actions

        # Use by default class value, else use given value
        cell_length = cell_length if cell_length is not None else self.cell_length
        angle_deviation = angle_deviation if angle_deviation is not None else self.angle_deviation
        tropism_sensitivity = tropism_sensitivity if tropism_sensitivity is not None else self.tropism_sensitivity
        tropism_max_bend = tropism_max_bend if tropism_max_bend is not None else self.tropism_max_bend

        # Link to calculator objects
        CellGeometryCalculator.__init__(self, length=cell_length, angle_deviation=angle_deviation, bend=bend)
        TropismCalculator.__init__(self, sensitivity=tropism_sensitivity, max_bend=tropism_max_bend)

    @action_timer.measure_decorator("GrowCell")
    def update(self, cell: Cell):
        new_cell = self._create_new_cell(cell)      # Create a new cell based on the position of the parent cell
        self._link_new_cell(cell, new_cell)         # Link new cell to the parent colony
        self.add_base()                             # Add base values for the new cell
        self._execute_all_actions(cell, new_cell)   # Execute all growth actions

    def _create_new_cell(self, parent_cell: Cell) -> Cell:
        """Creates a new cell based on parent cell."""
        tropism_bend = self.calc_tropism_bend(parent_cell)
        center, end, direction = self.calculate_new_cell_points(
            parent_cell,
            tropism_bend
        )
        return Cell(center, end, np.degrees(direction), parent=parent_cell, length=parent_cell.length)

    def _execute_all_actions(self, parent: Cell, child: Cell):
        """Execute the parent, new cell and intercellular actions."""
        for action in self.parent_actions:
            action.update(parent)
        for action in self.new_actions:
            action.update(child)
        for duo_action in self.relation_actions:
            duo_action.update(parent, child)

    @staticmethod
    def _link_new_cell(parent: Cell, child: Cell) -> None:
        """
        Links the child cell to the parent and the same colony as the parent.

        :param parent: The parent cell whose colony the child will join.
        :param child: The newly added cell that joins the parent's colony.
        """
        parent.link_child(child)
        Colony.instances[parent.colony_index].add_cell(child)


class AddDivIVA(Action):
    def __init__(self, polarisome_amount):
        self.amount = polarisome_amount

    @action_timer.measure_decorator("AddDivIVA")
    def update(self, cell: Cell):
        cell.DivIVA += self.amount


class CrowdingIndex(Action):
    crowding_steepness = 0
    spacing = 0

    def __init__(self, condition: Condition, alpha: float = 0):
        self.condition_index: int = condition.index     # The condition index that stores the crowding factor
        self.alpha: float = alpha                       # The strength value for the crowding factor intensity

    @action_timer.measure_decorator("CrowdingIndex")
    def update(self, cell: Cell) -> None:
        with action_timer.measure("CrowdingIndex: _get_valid_neighbours"):
            neighbour_indexes, distances = self._get_valid_neighbours(cell)

        with action_timer.measure("CrowdingIndex: calc_base_crowding_index"):
            crowding_values = self.calc_base_crowding_index(distances)

        with action_timer.measure("CrowdingIndex: update_value"):
            self.update_crowding(cell.index, neighbour_indexes, crowding_values, Cell.crowding_index_array.active)
            # Cell.crowding_index_array[neighbour_indexes] += crowding_values
            # cell.crowding_index += crowding_values.sum()

        with action_timer.measure("CrowdingIndex: _set_condition_factor"):
            self._set_condition_factor(cell, neighbour_indexes)

    @staticmethod
    @njit
    def update_crowding(cell_idx, neighbour_indexes, crowding_values, crowding_index_array):
        total = 0.0
        for i in range(len(neighbour_indexes)):
            idx = neighbour_indexes[i]
            val = crowding_values[i]
            crowding_index_array[idx] += val
            total += val

        crowding_index_array[cell_idx] += total

    def _get_valid_neighbours(self, cell: Cell) -> (np.ndarray, np.ndarray):
        neighbour_indexes: np.ndarray = Colony.get_all_neighbours(cell)
        points: np.ndarray = Cell.center_point_array.get_rows(neighbour_indexes)

        # diffs: np.ndarray = np.subtract(points, cell.center)
        # distances_squared: np.ndarray = np.einsum('ij,ij->i', diffs, diffs)
        #
        # distances_filter: np.ndarray = distances_squared <= self.space.partition_size ** 2
        # distances: np.ndarray = np.sqrt(distances_squared[distances_filter])
        # neighbour_filt: np.ndarray = neighbour_indexes[distances_filter]

        return self.compute_valid_neighbours(cell.center, neighbour_indexes, points, self.space.partition_size ** 2)
        #
        # return neighbour_filt, distances

    @staticmethod
    @njit
    def compute_valid_neighbours(cell_center, neighbour_indexes, neighbour_points, max_sq_dist):
        n = neighbour_points.shape[0]

        # Preallocate output with max possible size (worst case: all valid)
        valid_indexes = np.empty(n, dtype=neighbour_indexes.dtype)
        valid_distances = np.empty(n, dtype=neighbour_points.dtype)

        count = 0
        for i in range(n):
            dx = neighbour_points[i, 0] - cell_center[0]
            dy = neighbour_points[i, 1] - cell_center[1]
            dz = neighbour_points[i, 2] - cell_center[2]

            dist_sq = dx * dx + dy * dy + dz * dz

            if dist_sq <= max_sq_dist:
                valid_indexes[count] = neighbour_indexes[i]
                valid_distances[count] = np.sqrt(dist_sq)
                count += 1

        # Trim to actual size
        return valid_indexes[:count], valid_distances[:count]

    def _set_condition_factor(self, cell: Cell, neighbour_indexes):
        self.condition_factors[neighbour_indexes, self.condition_index] = self._calc_crowding_factor(Cell.crowding_index_array[neighbour_indexes])
        self.condition_factors[cell.index, self.condition_index] = self._calc_crowding_factor(Cell.crowding_index_array[cell.index])

    def _calc_crowding_factor(self, crowding: float or np.ndarray) -> float or np.ndarray:
        return 1 / (1 + (crowding * self.alpha))

    @classmethod
    def calc_base_crowding_index(cls, distances: np.ndarray) -> np.ndarray:
        """
        Calculates the crowding index.

        :param distances: array of distances of neighbouring cells.
        :return: crowding index values
        """
        return np.exp(-distances / cls.crowding_steepness) * cls.spacing

    @classmethod
    def calculate_query_size(cls, error_tolerance: float) -> float:
        """
        Calculates the cutoff distance when the crowding index value is less than error_tolerance_significance.
        Use this value for setting the space search radius.
        Must set a value for crowding_steepness beforehand.

        :param error_tolerance: The error tolerance value.
        :return: The cutoff distance based on the error tolerance.
        """
        return float(-np.log(error_tolerance / cls.spacing) * cls.crowding_steepness)


class Fragment(Action):
    def __init__(self, stump_state: State):
        self.stump_state: State = stump_state

    @action_timer.measure_decorator("Fragment")
    def update(self, cell: Cell):
        self.decouple_from_parent(cell)
        branch = self.form_branch(cell)
        cell.children = []

        old_colony: Colony = Colony.instances[cell.colony_index]
        branch_stump = Cell.create_root_cell(cell.center, cell.direction, self.stump_state)
        new_colony = Colony(branch_stump)

        old_colony.remove_branch(branch)
        new_colony.add_branch(branch)

    @staticmethod
    def decouple_from_parent(cell):
        cell.parent.children.remove(cell)

    def form_branch(self, cell: Cell) -> list[Cell]:
        branch = []
        for child in cell.children:
            branch += self.form_branch(child)
        return branch


class Transfer(DuoAction):
    def __init__(self, parameter: str, transfer_ratio=1):
        self.param = parameter
        self.ratio = transfer_ratio

    @action_timer.measure_decorator("Transfer")
    def update(self, donor_cell: Cell, recipient_cell: Cell):
        transfer_amount = getattr(donor_cell, self.param) * self.ratio
        self._transfer_param(donor_cell, -transfer_amount)
        self._transfer_param(recipient_cell, transfer_amount)

    def _transfer_param(self, cell: Cell, delta: float):
        setattr(cell, self.param, getattr(cell, self.param) + delta)


if __name__ == '__main__':
    from src.utils.benchmark_timer import Timer

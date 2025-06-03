from abc import ABC, abstractmethod
import numpy as np
from numba import njit
import math

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


class CollectValidNeighbours(Action):
    """Create a cash of all filtered neighbours"""
    neighbour_indexes_cache: np.ndarray = None
    neighbour_points_cache: np.ndarray = None
    distances_cache: np.ndarray = None

    @classmethod
    @action_timer.measure_decorator("CollectValidNeighbours")
    def update(cls, cell: Cell):
        with action_timer.measure("pickup data inds"):
            neighbour_indexes: np.ndarray = cls.get_all_neighbours(cell)
        with action_timer.measure("pickup data points"):
            # points: np.ndarray = Cell.center_point_array.get_points(neighbour_indexes)
            points: np.ndarray = Cell.center_point_array[neighbour_indexes]

        # Filter based on distance
        with action_timer.measure("big calc"):
            max_dist = cls.space.partition_size
            inds, dists = cls.compute_valid_neighbours(
                cell.center, neighbour_indexes, points, max_dist
            )

        with action_timer.measure("save to cash"):
            # Store the new data inside the cashes
            cls.neighbour_indexes_cache = inds
            cls.neighbour_points_cache = Cell.center_point_array[inds]
            cls.distances_cache = dists

    @staticmethod
    def get_all_neighbours(cell):
        return Colony.instances[cell.colony_index].cell_grid.query(cell.center)

    @staticmethod
    @action_timer.measure_decorator("compute_valid_neighbours")
    @njit(fastmath=True)
    def compute_valid_neighbours(cell_center: np.ndarray,
                                 neighbour_indexes: np.ndarray,
                                 neighbour_points: np.ndarray,
                                 max_dist: float) -> (np.ndarray, np.ndarray):
        """
        Filters and returns valid neighboring cells within a specified maximum distance
        from the target cell, along with their distances.

        :param cell_center: 2D coordinates of the target cell
        :param neighbour_indexes: Array of indices of potential valid neighbours
        :param neighbour_points: 2D coordinates of the potential neighbours
        :param max_dist: filter distance from target cell, exclude neighbour if distance is greater than the max
        :return: Array of valid neighbours, array of valid distances (arrays are coupled element wise)
        """
        # Allocate memory space
        n = neighbour_points.shape[0]
        valid_indexes = np.empty(n, dtype=neighbour_indexes.dtype)
        valid_distances = np.empty(n, dtype=neighbour_points.dtype)

        count = 0                                   # Total number of valid neighbours
        cx, cy = cell_center[0], cell_center[1]     # Cache cell values
        max_sq_dist = max_dist * max_dist
        for i in range(n):
            # Relative displacement
            dx = neighbour_points[i, 0] - cx
            dy = neighbour_points[i, 1] - cy

            # Distance squared (skip square root calculations for invalid distances)
            dist_squared = dx * dx + dy * dy

            # Filter if dist^2 <= max_dist^2 (same as dist <= max_dist)
            if dist_squared <= max_sq_dist:
                valid_indexes[count] = neighbour_indexes[i]
                valid_distances[count] = dist_squared ** 0.5
                count += 1

        # Trim to correct size
        return valid_indexes[:count], valid_distances[:count]

    @staticmethod
    @action_timer.measure_decorator("compute_distance")
    @njit(fastmath=True)
    def compute_distance(target_point: np.ndarray, neighbour_points: np.ndarray) -> np.ndarray:
        """
        Assumes that neighbours all already filtered

        :param target_point:
        :param neighbour_points:
        :return:
        """
        n = neighbour_points.shape[0]
        distances = np.empty(n, dtype=np.float64)

        tx, ty = target_point[0], target_point[1]
        for i in range(n):
            dx = neighbour_points[i, 0] - tx
            dy = neighbour_points[i, 1] - ty
            distance = (dx * dx + dy * dy) ** 0.5
            distances[i] = distance

        return distances


class CellGeometryCalculator:
    """Handles geometric calculations for new cell positions"""

    def __init__(self, length, angle_deviation=0, bend=0):
        self.length = length
        self.angle_deviation = np.radians(angle_deviation)
        self.bend = np.radians(bend)

    @action_timer.measure_decorator("calculate_new_cell_points")
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

    @action_timer.measure_decorator("spatial_calculations")
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

    @action_timer.measure_decorator("calc_tropism_bend")
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

        neighbour_points = CollectValidNeighbours.neighbour_points_cache

        left_sample, right_sample = self._get_sampler_points(cell)
        left_stimulus = self._calc_total_stimulus(neighbour_points, left_sample)
        right_stimulus = self._calc_total_stimulus(neighbour_points, right_sample)

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
    def _calc_total_stimulus(neighbour_points, sample_points):
        """Calculate the total crowding stimulus from a single sample point"""
        dists: np.ndarray = CollectValidNeighbours.compute_distance(sample_points, neighbour_points)
        return CrowdingIndex.calc_base_crowding_index(dists).sum()


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
                 get_neighbours=False,
                 cell_length=None, angle_deviation=None, bend=0,
                 tropism_sensitivity=None, tropism_max_bend=None):
        self.parent_actions = parent_cell_actions
        self.new_actions = new_cell_actions
        self.relation_actions = dou_actions
        self.get_neighbours: bool = get_neighbours

        # Use by default class value, else use given value
        cell_length = cell_length if cell_length is not None else self.cell_length
        angle_deviation = angle_deviation if angle_deviation is not None else self.angle_deviation
        tropism_sensitivity = tropism_sensitivity if tropism_sensitivity is not None else self.tropism_sensitivity
        tropism_max_bend = tropism_max_bend if tropism_max_bend is not None else self.tropism_max_bend

        # Link to subclasses
        CellGeometryCalculator.__init__(self, length=cell_length, angle_deviation=angle_deviation, bend=bend)
        TropismCalculator.__init__(self, sensitivity=tropism_sensitivity, max_bend=tropism_max_bend)

    @action_timer.measure_decorator("GrowCell")
    def update(self, cell: Cell):
        self.collect_neighbours(cell)               # Get all valid neighbours
        new_cell = self._create_new_cell(cell)      # Create a new cell based on the position of the parent cell
        self._link_new_cell(cell, new_cell)         # Link new cell to the parent colony
        self.add_base()                             # Add base values for the new cell
        self._execute_all_actions(cell, new_cell)   # Execute all growth actions

    def collect_neighbours(self, cell: Cell):
        if self.get_neighbours:
            CollectValidNeighbours.update(cell)

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

    def add_base(self):
        self.event_propensities.append(0)
        self.state_mask.append(0)
        self.condition_factors.append(0)


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

    def update(self, cell: Cell) -> None:
        """Add the crowding effect of a cell from its neighbors and itself."""
        neighbour_indexes = CollectValidNeighbours.neighbour_indexes_cache
        distances = CollectValidNeighbours.distances_cache

        crowding_values = self.calc_base_crowding_index(distances)
        self.add_crowding(cell.index, neighbour_indexes, crowding_values, Cell.crowding_index_array.active)
        self._set_condition_factor(cell, neighbour_indexes)

    @staticmethod
    @action_timer.measure_decorator("Add crowding")
    @njit
    def add_crowding(cell_idx, neighbour_indexes, crowding_values, crowding_index_array):
        total = 0.0
        for i, n_ind in enumerate(neighbour_indexes):
            val = crowding_values[i]
            crowding_index_array[n_ind] += val
            total += val
        crowding_index_array[cell_idx] += total

    def _calc_crowding_factor(self, crowding: float or np.ndarray) -> float or np.ndarray:
        return 1 / (1 + (crowding * self.alpha))

    def _set_condition_factor(self, cell: Cell, neighbour_indexes):
        # Set the crowding factor for the neighbours
        self.condition_factors[neighbour_indexes, self.condition_index] = self._calc_crowding_factor(
            Cell.crowding_index_array[neighbour_indexes])
        # Set the crowding factor for the target cell
        self.condition_factors[cell.index, self.condition_index] = self._calc_crowding_factor(
            Cell.crowding_index_array[cell.index])

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


# class CrowdingIndex(Action):
#     crowding_steepness = 0
#     spacing = 0
#
#     def __init__(self, condition: Condition, alpha: float = 0):
#         self.condition_index: int = condition.index     # The condition index that stores the crowding factor
#         self.alpha: float = alpha                       # The strength value for the crowding factor intensity
#
#     def update(self, cell: Cell) -> None:
#         """Add the crowding effect of a cell from its neighbors and itself."""
#         neighbour_indexes, distances = self.get_valid_neighbours(cell)
#         crowding_values = self.calc_base_crowding_index(distances)
#         self.add_crowding(cell.index, neighbour_indexes, crowding_values, Cell.crowding_index_array.active)
#         # self.remove_crowding(cell.index, neighbour_indexes, crowding_values, Cell.crowding_index_array.active)
#         self._set_condition_factor(cell, neighbour_indexes)
#
#     @staticmethod
#     @njit
#     def remove_crowding(cell_idx, neighbour_indexes, crowding_values, crowding_index_array):
#         total = 0.0
#         for i, n_ind in enumerate(neighbour_indexes):
#             val = crowding_values[i]
#             crowding_index_array[n_ind] -= val
#             total += val
#         crowding_index_array[cell_idx] -= total
#
#     @staticmethod
#     @action_timer.measure_decorator("Add crowding")
#     @njit
#     def add_crowding(cell_idx, neighbour_indexes, crowding_values, crowding_index_array):
#         total = 0.0
#         for i, n_ind in enumerate(neighbour_indexes):
#             val = crowding_values[i]
#             crowding_index_array[n_ind] += val
#             total += val
#         crowding_index_array[cell_idx] += total
#
#     @classmethod
#     def update_remove(cls, cell: Cell) -> None:
#         """Removes the crowding effect of a cell from its neighbors and itself."""
#         n_inds, dists = cls.get_valid_neighbours(cell)
#         crowding_values = cls.calc_base_crowding_index(dists)
#         cls.remove_crowding(
#             cell.index,
#             n_inds,
#             crowding_values,
#             Cell.crowding_index_array.active
#         )
#
#     @classmethod
#     @action_timer.measure_decorator("Crowding: get valid neighbours")
#     def get_valid_neighbours(cls, cell: Cell) -> tuple[np.ndarray, np.ndarray]:
#         neighbour_indexes: np.ndarray = Colony.get_all_neighbours(cell)
#         points: np.ndarray = Cell.center_point_array[neighbour_indexes]
#
#         # Filter the neighbours based on the distance
#         neighbour_indexes, dists = cls.compute_valid_neighbours(cell.center, neighbour_indexes, points, cls.space.partition_size)
#         return neighbour_indexes, dists
#
#     @staticmethod
#     @action_timer.measure_decorator("Valid neighbours")
#     @njit
#     def compute_valid_neighbours(cell_center: np.ndarray,
#                                  neighbour_indexes: np.ndarray, neighbour_points: np.ndarray,
#                                  max_dist: float):
#         n = neighbour_points.shape[0]
#         max_sq_dist = max_dist * max_dist
#
#         # Preallocate output with max possible size
#         valid_indexes: np.ndarray = np.empty(n, dtype=neighbour_indexes.dtype)
#         valid_distances: np.ndarray = np.empty(n, dtype=neighbour_points.dtype)
#
#         count: int = 0
#         cx, cy = cell_center[0], cell_center[1]
#         for i in range(n):
#             dx = neighbour_points[i, 0] - cx
#             dy = neighbour_points[i, 1] - cy
#             dist_squared = dx * dx + dy * dy
#
#             if dist_squared <= max_sq_dist:
#                 valid_indexes[count] = neighbour_indexes[i]
#                 valid_distances[count] = math.sqrt(dist_squared)
#                 count += 1
#
#         # Trim to actual size
#         return valid_indexes[:count], valid_distances[:count]
#
#     def _set_condition_factor(self, cell: Cell, neighbour_indexes):
#         self.condition_factors[neighbour_indexes, self.condition_index] = self._calc_crowding_factor(Cell.crowding_index_array[neighbour_indexes])
#         self.condition_factors[cell.index, self.condition_index] = self._calc_crowding_factor(Cell.crowding_index_array[cell.index])
#
#     def _calc_crowding_factor(self, crowding: float or np.ndarray) -> float or np.ndarray:
#         return 1 / (1 + (crowding * self.alpha))
#
#     @classmethod
#     def calc_base_crowding_index(cls, distances: np.ndarray) -> np.ndarray:
#         """
#         Calculates the crowding index.
#
#         :param distances: array of distances of neighbouring cells.
#         :return: crowding index values
#         """
#         return np.exp(-distances / cls.crowding_steepness) * cls.spacing
#
#     @classmethod
#     def calculate_query_size(cls, error_tolerance: float) -> float:
#         """
#         Calculates the cutoff distance when the crowding index value is less than error_tolerance_significance.
#         Use this value for setting the space search radius.
#         Must set a value for crowding_steepness beforehand.
#
#         :param error_tolerance: The error tolerance value.
#         :return: The cutoff distance based on the error tolerance.
#         """
#         return float(-np.log(error_tolerance / cls.spacing) * cls.crowding_steepness)


class Fragment(Action):
    def __init__(self, parent_reawaken_state: State):
        self.parent_reawaken = SwitchState(parent_reawaken_state)

    @action_timer.measure_decorator("Fragment")
    def update(self, cell: Cell):
        print(f"WE FRAGMENT {cell.index} / {Cell.total}")
        self.split_from_parent(cell)

        branch = self.form_branch(cell)
        print("all same colony", all([cell.colony_index == bcell.colony_index for bcell in branch]))
        print(f"branch size {len(branch)}")

        self.full_remove_branch_from_colony(cell, branch)

        new_colony = Colony(cell)
        new_colony.add_branch(branch[1:])
        print("min new", min(Cell.crowding_index_array[new_colony.cell_indexes]))
        print("max new", max(Cell.crowding_index_array[new_colony.cell_indexes]))

    def full_remove_branch_from_colony(self, cell: Cell, branch):
        # Get the colony
        old_colony: Colony = Colony.instances[cell.colony_index]

        crowding_col = Cell.crowding_index_array[old_colony.cell_indexes]
        min_crowding = min(crowding_col)
        max_crowding = max(crowding_col)

        print("min old pre", min_crowding)
        print("max old pre", max_crowding)

        # Remove the crowding values from the old colony
        self.remove_crowding(branch)

        # Decouple cell indexes from the colony
        old_colony.remove_branch(branch)

        print("min old pos", min(Cell.crowding_index_array[old_colony.cell_indexes]))
        print("max old pos", max(Cell.crowding_index_array[old_colony.cell_indexes]))

    def split_from_parent(self, cell):
        """
        Decouple the cell from the parent.
        Reawaken the parent so it continues growing.
        """
        parent = cell.parent
        parent.children.remove(cell)
        # self.parent_reawaken.update(parent)
        cell.parent = None

    def form_branch(self, cell: Cell) -> list[Cell]:
        branch = [cell]
        for child in cell.children:
            branch.extend(self.form_branch(child))
        return branch

    def remove_crowding(self, branch: list[Cell]):
        branch_index_set = {cell.index for cell in branch}  # Fast lookup
        for branch_cell in branch:
            n_inds, dists = CrowdingIndex.get_valid_neighbours(branch_cell)

            # Create a boolean mask for neighbors NOT in the branch
            mask = ~np.isin(n_inds, list(branch_index_set))

            # Apply the mask to filter neighbors and distances
            n_inds_filtered = n_inds[mask]
            dists_filtered = dists[mask]

            crowding_values = CrowdingIndex.calc_base_crowding_index(dists_filtered)

            CrowdingIndex.remove_crowding(
                branch_cell.index,
                n_inds_filtered,
                crowding_values,
                Cell.crowding_index_array.active
            )

    # def remove_crowding(self, branch: list[Cell]):
    #     for branch_cell in branch:
    #         n_inds, dists = CrowdingIndex.get_valid_neighbours(branch_cell)
    #         crowding_values = CrowdingIndex.calc_base_crowding_index(dists)
    #         CrowdingIndex.remove_crowding(
    #             branch_cell.index,
    #             n_inds,
    #             crowding_values,
    #             Cell.crowding_index_array.active
    #         )

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

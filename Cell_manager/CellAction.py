from abc import ABC, abstractmethod
import numpy as np

from utils.DynamicArray import Dynamic2DArray
from Space_manager.SpatialHashing import SpatialHashing
from Cell_manager.Cell import Cell, Root
from Cell_manager.Colony import Colony
from Event_manager.State import State
from Event_manager.Condition import Condition


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

    def update(self, cell: Cell):
        # print(cell.state.event_mask)
        cell.state = self.new_state
        State.cell_mask_array[cell.index] = cell.state.event_mask


class TropismCalculator:
    """Handles tropism-related calculations"""

    def __init__(self, sensitivity=0, max_bend=0):
        self.alpha = sensitivity
        self.max_bend = np.radians(max_bend)
        self.sampler_distance = 1e-6
        self.half_pi = np.pi / 2

    def update(self, cell: Cell) -> float:
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
    def _calc_total_stimulus(neighbour_indexes, sample_point):
        """Calculate the total crowding stimulus from a single sample point"""
        points = Cell.center_point_array[neighbour_indexes]
        dists = Action.space.calc_distances(points, sample_point)
        filtered_dists = Action.space.filter_distances(dists)
        return CrowdingIndex.calc_crowding_index(filtered_dists).sum()


class CellGeometryCalculator:
    """Handles geometric calculations for new cell positions"""

    def __init__(self, length, angle_deviation=0, bend=0):
        self.length = length
        self.angle_deviation = np.radians(angle_deviation)
        self.bend = np.radians(bend)

    def update(self, cell: Cell):
        return

    def calculate_new_cell_points(self, parent_cell: Cell, tropism_bend: float = 0) -> (tuple[int, int], tuple[int, int], float):
        """
        Calculate new cell's center, end, and direction.

        :param parent_cell: Used to copy the direction
        :param tropism_bend: Bend towards nutrients, away from crowded spaces (in radians)
        :return: Cell center point, Cell end point, direction (in radians)
        """
        noise = np.random.normal(0, self.angle_deviation)
        random_bend = np.random.choice([self.bend, -self.bend])
        new_direction = parent_cell.direction + random_bend + tropism_bend + noise

        return self.spatial_calculations(parent_cell, new_direction)

    def spatial_calculations(self, cell: Cell, new_direction: float) -> (tuple[int, int], tuple[int, int], float):
        x, y = cell.end
        dx = self.length * np.sin(new_direction)
        dy = self.length * np.cos(new_direction)

        new_center = (x + (dx / 2), y + (dy / 2))
        new_end = (x + dx, y + dy)

        return new_center, new_end, new_direction


class GrowCell(Action):
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

        # Can overwrite the class default
        cell_length = cell_length if cell_length is not None else self.cell_length
        angle_deviation = angle_deviation if angle_deviation is not None else self.angle_deviation
        tropism_sensitivity = tropism_sensitivity if tropism_sensitivity is not None else self.tropism_sensitivity
        tropism_max_bend = tropism_max_bend if tropism_max_bend is not None else self.tropism_max_bend

        # Link to calculator objects
        self.geometry = CellGeometryCalculator(cell_length, angle_deviation, bend)
        self.tropism = TropismCalculator(tropism_sensitivity, tropism_max_bend)

    def update(self, cell: Cell):
        new_cell = self._create_new_cell(cell)      # Create a new cell based on the position of the parent cell
        self._link_new_cell(cell, new_cell)         # Link new cell to the parent colony
        self.add_base()                             # Add base values for the new cell
        self._execute_all_actions(cell, new_cell)   # Execute all growth actions

    def _create_new_cell(self, parent_cell: Cell) -> Cell:
        """Creates a new cell based on parent cell."""
        tropism_bend = self.tropism.update(parent_cell)
        center, end, direction = self.geometry.calculate_new_cell_points(
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
        Colony.colonies[parent.colony_index].add_cell(child)


class AddDivIVA(Action):
    def __init__(self, polarisome_amount):
        self.amount = polarisome_amount

    def update(self, cell: Cell):
        cell.DivIVA += self.amount


class CrowdingIndex(Action):
    k = 0

    @classmethod
    def calculate_query_size(cls, error_tolerance_significance: float) -> float:
        """
        Calculates the cutoff distance when the crowding index value is less than 10**-error_tolerance_significance.
        Use this value for setting the space search radius.
        Must set a value for k beforehand.

        :param error_tolerance_significance: The error tolerance significant value for base 10.
        :return: The cutoff distance based on the error tolerance.
        """
        return float(error_tolerance_significance * np.e * cls.k * np.log10(10))

    def __init__(self, condition: Condition, alpha: float = 0):
        self.condition_index = condition.index
        self.alpha = alpha

    def update(self, cell: Cell) -> None:
        neighbour_indexes, distances = self._get_valid_neighbours(cell)
        if len(neighbour_indexes) == 0:
            return
        crowding_indexes = self.calc_crowding_index(distances)
        self._set_crowding_index(cell, neighbour_indexes, crowding_indexes)
        self._set_condition_factor(cell, neighbour_indexes, crowding_indexes)

    def _get_valid_neighbours(self, cell: Cell) -> (np.ndarray, np.ndarray):
        neighbour_indexes = np.array(Colony.get_all_neighbours(cell))
        if len(neighbour_indexes) == 0:
            return

        points = Cell.center_point_array[neighbour_indexes]
        distances = self.space.calc_distances(points, cell.center)

        distances_filter = distances <= self.space.partition_size
        return neighbour_indexes[distances_filter], distances[distances_filter]

    @staticmethod
    def _set_crowding_index(cell: Cell, neighbour_indexes, crowding_indexes) -> None:
        cell.crowding_index += crowding_indexes.sum()
        Cell.batch_update_crowding(neighbour_indexes, crowding_indexes)

    def _set_condition_factor(self, cell: Cell, neighbour_indexes, crowding_indexes):
        crowding_indexes *= self.alpha
        crowding_factor = self._crowding_factor(crowding_indexes)
        self.condition_factors[neighbour_indexes, self.condition_index] = crowding_factor
        self.condition_factors[cell.index, self.condition_index] = self._crowding_factor(cell.crowding_index)

    @staticmethod
    def _crowding_factor(crowding):
        return 1 / (1 + crowding)

    @classmethod
    def calc_crowding_index(cls, distances: np.ndarray) -> np.ndarray:
        """
        Calculates the crowding index.

        :param distances: array of distances of neighbouring cells.
        :return: influence values
        """
        return np.exp(-distances / (np.e * cls.k))


# class CrowdingIndex(Action):
#     k = 0
#
#     @classmethod
#     def calculate_query_size(cls, error_tolerance_significance: float) -> float:
#         """
#         Calculates the cutoff distance when the crowding index value is less than 10**-error_tolerance_significance.
#         Use this value for setting the space search radius.
#         Must set a value for k beforehand.
#
#         :param error_tolerance_significance: The error tolerance significant value for base 10.
#         :return: The cutoff distance based on the error tolerance.
#         """
#         return float(error_tolerance_significance * np.e * cls.k * np.log10(10))
#
#     def __init__(self, condition: Condition, alpha: float = 0):
#         self.condition_index = condition.index
#         self.alpha = alpha
#
#     def update(self, cell: Cell):
#         # Query all neighbours in the local area (radius = k)
#         neighbour_indexes = np.array(Colony.get_all_neighbours(cell))
#
#         # If no neighbours then can skip all the calculations
#         if len(neighbour_indexes) == 0:
#             return
#
#         # Get the relevant center points (to have a rough estimation of the entire cell position)
#         points = Cell.center_point_array[neighbour_indexes]
#
#         # Calculate all the distances between the neighbour points and the current cell
#         dists = self.space.calc_distances(points, cell.center)
#
#         # Filter out points that are beyond the searching distance (they get ignored in following calculations)
#         # valid_dists = self.space.filter_distances(dists)
#         valid_mask = dists <= self.space.partition_size
#         valid_indices = neighbour_indexes[valid_mask]
#         valid_dists = dists[valid_mask]
#
#         # Calculate the crowding index based on the distances
#         crowding_indexes = self.calc_crowding_index(valid_dists)
#
#         # Update all neighbours with newly added influence
#         cell.crowding_index += crowding_indexes.sum()
#         Cell.batch_update_influences(valid_indices, crowding_indexes)
#
#         # Convert the crowding to a factor
#         crowding_indexes *= self.alpha
#         crowding_factor = 1 / (1 + crowding_indexes)
#         self.condition_factors[valid_indices, self.condition_index] = crowding_factor
#         self.condition_factors[cell.index, self.condition_index] = 1 / (1 + cell.crowding_index)
#
#     @classmethod
#     def calc_crowding_index(cls, distances):
#         """
#         Calculates the crowding index.
#
#         :param distances: array of distances of neighbouring cells.
#         :return: influence values
#         """
#         return np.exp(-distances / (np.e * cls.k))


class Fragment(Action):
    def __init__(self, stump_state: State):
        self.stump_state: State = stump_state

    def update(self, cell: Cell):
        self.decouple_from_parent(cell)
        branch = self.form_branch(cell)
        cell.children = []

        old_colony: Colony = Colony.colonies[cell.colony_index]
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

    def update(self, donor_cell: Cell, recipient_cell: Cell):
        transfer_amount = getattr(donor_cell, self.param) * self.ratio
        self._transfer_param(donor_cell, -transfer_amount)
        self._transfer_param(recipient_cell, transfer_amount)

    def _transfer_param(self, cell: Cell, delta: float):
        setattr(cell, self.param, getattr(cell, self.param) + delta)


if __name__ == '__main__':
    from utils.BenchmarkTimer import Timer

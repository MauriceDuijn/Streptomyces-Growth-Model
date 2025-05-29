import warnings
import numpy as np
from itertools import product

from src.utils import DynamicArray
from src.algorithm.cell_based.cell import Cell


class SpacePartition:
    """
    SpacePartition holds inserted cells based on their cell center point.
    The maximum point within the space is 0 <= (x and y) < space size.
    """
    def __init__(self, space_size, query_size):
        self.size = space_size
        self.searching_distance = query_size

        # How many partitions are made in the square sized space
        self.number_of_partitions = int(space_size / query_size)

        if self.number_of_partitions < 3:
            warnings.warn("\nThe number of partitions is less then 3, this can cause incorrect query results.", UserWarning)
            self.number_of_partitions = 1

        # The exact length of a partition
        self.partition_modulo = space_size / self.number_of_partitions

        # Create grid based on the given settings
        self.cell_grid = [[DynamicArray(data_type=np.int64)
                           for x_partition in range(self.number_of_partitions)]
                          for y_partition in range(self.number_of_partitions)]

        self.offsets = np.array(list(product([-1, 0, 1], repeat=2))) if self.number_of_partitions != 1 else 0

    def point_to_partition_index(self, point):
        """
        Convert the given point to the corresponding partition index.

        :param point: Input point that gets translated to corresponding partition index.
        :return: (x partition index, y partition index)
        """
        return (point // self.partition_modulo).astype(np.int64)

    def insert(self, cell: Cell):
        """
        Insert a cell based on it's center point in the corresponding partition.

        :param cell: Input cell that gets stored inside the grid.
        """
        partition_index = self.point_to_partition_index(cell.center)

        self.cell_grid[partition_index[1]][partition_index[0]].append(cell.index)

    def wrap(self, point):
        """
        Ensures that given point stays in the range of (0, self.size)

        :param point: xy position
        :return: wrapped point
        """
        return (point[0] + self.size) % self.size, (point[1] + self.size) % self.size

    def query(self, point):
        """
        Get all cells of all neighbouring 3 x 3 partitions.
        [broken]

        :param point: Input target point.
        :return:
        """
        partition_index = self.point_to_partition_index(point)
        neighbor_partition_indices = (partition_index + self.offsets) % self.number_of_partitions
        neighbors = [self.cell_grid[y][x].active
                     for x, y in zip(neighbor_partition_indices[:, 0],
                                     neighbor_partition_indices[:, 1])]

        return np.concatenate(neighbors)

    @staticmethod
    def get_all_points():
        return Cell.center_point_array.active

    def get_random_point(self):
        """
        Get a random point that is within the current space [0, space_size).
        It includes 0 up until the set space size (so it's exclusive of the exact space size and above).

        :return: A single random point that is within the borders.
        """
        return np.random.uniform(0, self.size, size=(1, 2))

    def calc_distances(self, neighbour_positions, target):
        """
        Calculates all the distances of given neighbours points based on the input target point.

        :param neighbour_positions: Collection of neighbouring cell points, used for distance calculation.
        :param target: Point where all neighbouring points are used for distance calculation.
        :return: All distances of neighbouring cells relative to the target point.
        """
        # # Get the absolute difference of x and y based on the given target
        # absolute_difference = abs(neighbour_positions - target)
        #
        # # Get the smallest difference of the wrapped space
        # delta_difference = np.minimum(absolute_difference, self.size - absolute_difference)

        # Calculate distance on formatted points
        dists = np.linalg.norm(neighbour_positions - target, axis=1)

        return dists

    def filter_distances(self, distances):
        return distances[distances < self.searching_distance]

    def print_partition_sizes(self):
        for row in self.cell_grid:
            print("\t".join(str(len(partition)) for partition in row))

    @property
    def center(self):
        return self.size / 2, self.size / 2


if __name__ == '__main__':
    from src.utils import Timer

    timer = Timer()
    space = SpacePartition(100, 10)

    for i in range(10_000):
        with timer.measure("generate random point"):
            r_point = space.get_random_point()

        with timer.measure("create new cell"):
            new_cell = Cell(r_point, r_point)

        with timer.measure("cell insertion"):
            space.insert(new_cell)

    space.print_partition_sizes()
    timer.print_times()

    timer_2 = Timer()

    for i in range(1_000):
        with timer_2.measure("generate random point"):
            r_point = space.get_random_point()

        with timer_2.measure("query"):
            neighbours = space.query(r_point)

        with timer_2.measure("query to points"):
            neighbour_points = Cell.center_point_array[neighbours]

        with timer_2.measure("distance calc query"):
            query_dists = space.calc_distances(neighbour_points, r_point)

        with timer_2.measure("distance calc all"):
            all_dists = space.calc_distances(space.get_all_points(), r_point)

        with timer_2.measure("distance filter sum check"):
            query_sum = space.filter_distances(query_dists).sum()
            all_sum = space.filter_distances(all_dists).sum()
            if np.isclose(query_sum, all_sum):
                # The values are significantly the same
                pass
            else:
                print(query_sum, all_sum, "MISS")

    timer_2.print_times()





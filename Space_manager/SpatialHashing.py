import numpy as np
from Cell_manager.Cell import Cell


class SpatialHashing:
    partition_size: float = 0
    neighbour_offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]

    def __init__(self):
        self.grid: dict[tuple[int, int], list[int]] = {}

    def insert(self, cell: Cell) -> None:
        """
        Insert the cell index into the colony grid.

        1. Convert coördinate into a partition key
        2. Link cell index to the key (create a new list if key doesn't excist)

        :param cell: Cell to insert
        """
        cell_key: tuple[int, int] = self.get_cell_key(cell.center)
        self.grid.setdefault(cell_key, []).append(cell.index)

    def get_cell_key(self, point: tuple[float, float]) -> tuple[int, int]:
        """Convert coordinates to partition index."""
        x, y = point
        return int(x // self.partition_size), int(y // self.partition_size)

    def query(self, point: tuple[float, float]) -> np.ndarray:
        """
        Find all the cell indexes located within a 3x3 grid of partitions
        surrounding given point.

        :param point: Coördinate used for 3x3 grid lookup.
        :return: All cell indexes in neighbouring partitions.
        """
        x_key, y_key = self.get_cell_key(point)
        neighbours = []
        for dx, dy in self.neighbour_offsets:
            if grid_cell := self.grid.get((x_key + dx, y_key + dy)):
                neighbours.extend(grid_cell)
        return np.asarray(neighbours, dtype=np.int64)

    @classmethod
    def filter_distances(cls, distances: np.ndarray) -> np.ndarray:
        return distances[distances <= cls.partition_size]

    @classmethod
    def filter_distances_batch(cls, distances_nd: np.ndarray) -> list:
        return [distances[distances <= cls.partition_size] for distances in distances_nd]

    @classmethod
    def filter_distances_squared(cls, distances_squared: np.ndarray) -> np.ndarray:
        return distances_squared[distances_squared <= cls.partition_size ** 2]

    @classmethod
    def calc_distances(cls, neighbour_positions: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.linalg.norm(neighbour_positions - target, axis=1)

    @classmethod
    def calc_distances_multi(cls, neighbour_positions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return np.linalg.norm(neighbour_positions - targets[np.newaxis, :], axis=1)

    @classmethod
    def calc_distances_squared(cls, neighbour_positions: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.sum(np.square(neighbour_positions - target), axis=1)
        # dx = neighbour_positions[:, 0] - target[0]
        # dy = neighbour_positions[:, 1] - target[1]
        # return dx * dx + dy * dy


if __name__ == '__main__':
    from time import perf_counter
    space_size = 100
    sample_size = 100000
    n_tests = 100
    SpatialHashing.partition_size = space_size / 100
    space = SpatialHashing()

    start = perf_counter()
    for i in range(sample_size):
        random_point = np.random.uniform(0, space_size, size=(1, 2))
        space.insert(Cell(random_point, random_point, 0))
    print(f"insert {sample_size} items: ", perf_counter() - start)

    all = 0
    query = 0
    for i in range(n_tests):
        # New test cell
        random_point = np.random.uniform(0, space_size, size=(1, 2))
        test = Cell(random_point, random_point, 0)

        # dist check all
        start = perf_counter()
        dists = np.linalg.norm(Cell.center_point_array.active - test.center, axis=1)
        # print("all", dists.sum())
        print(space.filter_distances(dists).sum())
        all += perf_counter() - start

        # dist check via query
        start = perf_counter()
        result = Cell.center_point_array[space.query(test.center), :]
        dists = np.linalg.norm(result - test.center, axis=1)
        # print("query", dists.sum())
        print(space.filter_distances(dists).sum())
        query += perf_counter() - start

    print(all, query)

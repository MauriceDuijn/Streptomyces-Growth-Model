import numpy as np
from numba import njit


class DynamicArray:
    resize_factor: float = 2    # Expansion factor when resizing

    def __init__(self, capacity: int = 1000, data_type: np.dtype = np.float64):
        self.capacity: int = capacity
        self.row_size: int = 0
        self.dtype = data_type
        self.arr: np.ndarray = self.make_empty_array()

    def __repr__(self):
        return str(self.active)

    def __len__(self):
        return self.row_size

    def __getitem__(self, index):
        return self.arr[:self.row_size][index]

    def __setitem__(self, index, value):
        self.arr[:self.row_size][index] = value

    def __mul__(self, other):
        return self.active * other

    def __pow__(self, power):
        return self.active ** power

    @property
    def active(self) -> np.ndarray:
        return self.arr[:self.row_size]

    @active.setter
    def active(self, value):
        self.arr[:self.row_size] = value

    def make_empty_array(self):
        return np.zeros(self.capacity, dtype=self.dtype)

    def update_index(self, index: int, data: np.dtype) -> None:
        self.arr[index] = data

    def sum(self):
        return self.active.sum()

    def append(self, entry) -> None:
        # Check if next entry is outside the capacity bounds
        if self.row_size == self.capacity:
            self.resize()   # Resize the array

        self.update_index(self.row_size, entry)     # Add entry to end of the array
        self.row_size += 1                          # Update current row_size

    def resize(self) -> None:
        """Smartly resize the array by allocating new capacity in one operation."""
        new_capacity = int(np.ceil(self.capacity * self.resize_factor))
        self.capacity = new_capacity
        new_arr = self.make_empty_array()
        new_arr[:self.arr.size] = self.arr  # Copy existing repeat_data
        self.arr = new_arr

    def batch_remove(self, values: list):
        """Remove multiple items from the array."""
        self.arr = self.active[~np.isin(self.active, values)]
        self.row_size = self.arr.size
        self.capacity = self.arr.size

    @classmethod
    def load_data(cls, values: np.ndarray):
        if values.ndim != 1:
            raise ValueError(f"Given array of dimensions {values.ndim} doesn't fit in a 1D Dynamic2DArray.")

        darr = cls(values.shape[0], values.dtype)
        darr.arr = values
        darr.row_size = values.shape[0]

        return darr


class Dynamic2DArray(DynamicArray):
    def __init__(self, capacity_rows=1000, capacity_columns=0, data_type=np.float64):
        self.crows = capacity_rows
        self.ccols = capacity_columns
        super().__init__(capacity_rows, data_type)
        self.row_size = 0

    def make_empty_array(self):
        return np.zeros((self.crows, self.ccols), dtype=self.dtype)

    @classmethod
    def load_data(cls, values: np.ndarray):
        if values.ndim != 2:
            raise ValueError(f"Given array of dimensions {values.ndim} doesn't fit in a 2D Dynamic2DArray.")

        darr = cls(values.shape[0], values.shape[1], values.dtype)
        darr.arr = values
        darr.row_size = values.shape[0]

        return darr

    @property
    def size(self):
        return self.row_size * self.ccols

    @property
    def ndim(self):
        return self.row_size, self.ccols

    @property
    def active(self) -> np.ndarray:
        return self.arr[:self.row_size, :]

    def add_column(self):
        """
        Add an extra column to the array with base value 0.
        """
        self.ccols += 1
        new_arr = np.zeros((self.capacity, self.ccols), dtype=self.arr.dtype)
        new_arr[:, :self.ccols - 1] = self.arr
        self.arr = new_arr

    def update_row(self, row_index: int, data: np.ndarray):
        """
        Update a specific row in the event matrix with new values.

        :param row_index: Index of the row to update.
        :param data: New row repeat_data, must match the number of columns.
        """
        self.arr[row_index, :] = data

    def update_col(self, col_index: int, data: np.ndarray) -> None:
        """
        Update a specific column in the event matrix with new values.

        :param col_index: Index of the column to update.
        :param data: New column repeat_data, must match the number of active rows.
        """
        self.active[:, col_index] = data

    def append(self, entry) -> None:
        # Check if a resize is needed
        if self.row_size == self.crows:
            self.resize()

        # Update the row with the current index
        self.update_row(self.row_size, entry)

        # Update the new row size
        self.row_size += 1

    def resize(self) -> None:
        self.crows = int(np.ceil(self.crows * self.resize_factor))
        new_arr = np.zeros((self.crows, self.ccols), dtype=self.arr.dtype)
        new_arr[:self.row_size, :] = self.arr[:self.row_size, :]
        self.arr = new_arr

    def get_points(self, indexes):
        return self.get_points_njit(self.active, indexes)

    @staticmethod
    @njit
    def get_points_njit(darr: np.ndarray, indexes: np.ndarray):
        n = indexes.size
        points = np.empty((n, darr.shape[1]), dtype=darr.dtype)
        for i in range(n):
            points[i] = darr[indexes[i]]
        return points



# if __name__ == '__main__':
#     darr = DynamicArray()
#
#     print(darr.active)
#     for i in range(100):
#         darr.append(i)
#
#     print(darr.active)
#
#     d2arr = Dynamic2DArray()
#     d2arr.append(1)
#     d2arr.add_column()
#     print(d2arr.active)
#     for i in range(10):
#         d2arr.append((0, i + 1))
#
#     print(d2arr)
#     print(d2arr[3, 1])

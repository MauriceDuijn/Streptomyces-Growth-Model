from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

from Cell_manager.Cell import Cell
from utils.DynamicArray import DynamicArray, Dynamic2DArray


class Animator(ABC):
    @abstractmethod
    def initialize(self):
        """Setup for initial conditions"""
        pass

    @abstractmethod
    def save_snapshot(self):
        """Saves a snapshot during the simulation run"""
        pass

    @abstractmethod
    def update(self, frame: int):
        """Update the frame of the animation"""
        pass

    @abstractmethod
    def render(self):
        """Create the full animation"""
        pass


@dataclass
class Snapshot:
    cell_positions: np.ndarray      # Size(N_cells, 2)
    parameter_value: np.ndarray     # Size(N_cells, 1)


class CellGrowthAnimator(Animator):
    dpi = 300

    def __init__(self, space, cells, cell_parameter,
                 end_time, fps=1,
                 dot_size=0.1,
                 color_map="Spectral"):
        self.space_size = space.size
        self.cells = cells
        self.dot_size = dot_size

        # Snapshot settings
        self.position_array: Dynamic2DArray = Cell.center_point_array
        self.param_array: DynamicArray = getattr(Cell, f"{cell_parameter}_array")
        self.param_max: float = 0
        self.snapshots: list[Snapshot] = []

        # Frame settings
        self.end_time = end_time
        self.fps = fps
        self.sec_per_frame = 1.0 / fps
        self.next_frame = 0.0
        self.total_frames = end_time * fps
        print(f"[Total frames: {self.total_frames}]")

        # Set up plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, space.size)
        self.ax.set_ylim(0, space.size)

        # Initialize scatter with dummy point (removed later)
        self.scatter = self.ax.scatter([], [], c=[], s=dot_size,
                                       cmap=color_map, linewidths=0)
        self.cbar = self.fig.colorbar(self.scatter, ax=self.ax)
        colorbar_title = cell_parameter.replace('_', ' ').title()
        self.cbar.set_label(colorbar_title)
        self.max_label = self.ax.text(
            0.02, 0.95, '',
            transform=self.ax.transAxes
        )

        self.initialize()

    def initialize(self):
        """Initialize the spore cells"""
        self.scatter.set_offsets(Cell.center_point_array.active)
        self.save_snapshot()
        return self.scatter,

    def snapshot_schedule(self, current_time):
        """
        Store a snapshot if enough time has passed.
        When there are multiple frame updates without a simulation update,
        then multiple copies of the same snapshot are stored.
        """
        while current_time > self.next_frame:
            self.save_snapshot()
            self.next_frame += self.sec_per_frame

    def save_snapshot(self):
        """Stores the current run data as a snapshot, stored snapshots are used when rendering an animation"""
        self.param_max = max(self.param_max, max(self.param_array.active))
        sorted_indexes = np.argsort(self.param_array.active)
        self.snapshots.append(
            Snapshot(self.position_array.active[sorted_indexes], self.param_array.active[sorted_indexes])
        )

    def update(self, frame):
        """Update plot with cells up to the current frame."""
        snapshot = self.snapshots[frame]
        self.scatter.set_offsets(snapshot.cell_positions)
        self.scatter.set_array(snapshot.parameter_value)
        self.max_label.set_text(f"Max: {snapshot.parameter_value[-1]:.3f}")
        return self.scatter,

    def render(self, save_path=None, overwrite=False, speed=1):
        """
        Render the full growth animation.

        :param speed: How fast the animation runs, 1x speed equals 1 second per time unit.
        :param save_path: Filename (e.g., "growth.mp4") to save in 'utils/animations/'.
        :param overwrite: If current file name already exists, overwrite with new animation.
        """
        self.scatter.set_clim(0, self.param_max)

        anim = FuncAnimation(
            self.fig,
            self.update,
            init_func=self.initialize,
            frames=len(self.snapshots),
            interval=self.sec_per_frame * 1000 * speed,     # Seconds to milliseconds times the speed
            blit=True
        )

        if save_path:
            anim_dir = Path(__file__).parent / "animations"
            anim_dir.mkdir(exist_ok=True)
            full_path = anim_dir / save_path

            if overwrite and full_path.exists():
                full_path.unlink()  # Delete existing file
            elif full_path.exists():
                raise RuntimeError("File already exists, use overwrite=True to overwrite or delete existing file.")

            anim.save(full_path, writer='ffmpeg', dpi=self.dpi)
            print(f"✅ Animation saved to: {full_path}")

        else:
            plt.show()

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

from src.algorithm.cell_based.cell import Cell
from src.algorithm.cell_based.colony import Colony


class ColonyPlotter:
    dpi = 300

    def __init__(self, dot_size=3, linewidth=0):
        self.xmin = 0
        self.xmax = 100
        self.ymin = 0
        self.ymax = 100

        # Plot settings
        self.dotsize = dot_size * 72 / self.dpi
        self.linewidth = linewidth

        # Temporary repeat_data holder
        self.fig = None
        self.ax = None
        self.scatter = None
        self.cb = None

    def plot_cells_2D_colony(self, colony: Colony, parameter: str = None, cmap: str = "Spectral",
                             sort_values: bool = False, with_hull: bool = False):
        self.fig, self.ax = plt.subplots(dpi=self.dpi)
        self.ax.set_title("Colony structure")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        # Cell points
        points = colony.cell_points

        if parameter:
            # With parameter, add colorbar
            self._plot_paremeter(colony, points, parameter, sort_values, cmap)
        else:
            # No parameter, plot structure
            self._plot_simple(points)

        if with_hull:
            self._add_hull(points)

        # Adjust plot appearance
        self.ax.axis("equal")
        plt.show()

    def _plot_paremeter(self, colony: Colony, points: np.ndarray, parameter: str,
                        sort_values: bool, cmap: str):
        values = getattr(Cell, f"{parameter}_array").active[colony.cell_indexes]

        if sort_values:
            # Sort from the lowest value to highest
            sort_indexes = np.argsort(values)
            points = points[sort_indexes]
            values = values[sort_indexes]

        # Create scatter plot and store reference
        self.scatter = self.ax.scatter(
            points[:, 0], points[:, 1],
            c=values, cmap=cmap,
            s=self.dotsize, linewidths=self.linewidth
        )

        # Add colorbar only when parameter is provided
        self.cb = self.fig.colorbar(self.scatter, ax=self.ax)
        self.cb.set_label(parameter.replace('_', ' ').title())

    def _plot_simple(self, points: np.ndarray):
        self.ax.scatter(
            points[:, 0], points[:, 1],
            c='black',
            s=self.dotsize, linewidths=self.linewidth
        )

    def _add_hull(self, points: np.ndarray):
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        closed_hull = np.concatenate([hull_points, hull_points[:1]])
        self.ax.plot(closed_hull[:, 0], closed_hull[:, 1], 'r-', lw=2, label='Convex Hull')

    @staticmethod
    def plot_cells_3D_crowding_index():
        """[concept]"""
        x_data = Cell.center_point_array[1:, 0]
        y_data = Cell.center_point_array[1:, 1]
        crowding_index = Cell.crowding_index_array.active[1:]
        crowding_index += 1
        crowding_index **= -1

        # Create the 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x_data,
            y=y_data,
            z=crowding_index,
            mode='markers',
            marker=dict(
                size=6,
                color=crowding_index,   # Color based on crowding
                colorscale='RdYlGn_r',    # Red to Green colormap
                colorbar=dict(title='Crowding Index')
            )
        )])

        # Set the layout
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Crowding Index'
            ),
            title='3D Cell Crowding Index Plot'
        )

        # Show the interactive plot
        fig.show()

import os
import shutil
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import numpy as np

from Cell_manager.Cell import Cell
from Cell_manager.Colony import Colony
from Event_manager.Condition import Condition


class Plotter:
    dpi = 300

    def __init__(self, dot_size=3):
        self.xmin = 0
        self.xmax = 100
        self.ymin = 0
        self.ymax = 100

        # Plot settings
        self.dotsize = dot_size * 72 / self.dpi

        # Temporary data holder
        self.fig = None
        self.ax = None
        self.scatter = None
        self.cb = None

    def set_placeholder(self):
        self.fig, self.ax = plt.subplots(dpi=self.dpi)
        self.scatter = self.ax.scatter([], [], c=[], s=self.dotsize, cmap='Spectral', linewidths=0)
        self.cb = self.fig.colorbar(self.scatter, ax=self.ax)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)

    def plot_cells_2D(self, parameter: str = None, cmap: str = None,
                      sort_values: bool = False):
        # Copy template
        self.set_placeholder()

        # Change color map
        if cmap is not None:
            self.scatter.set_cmap(cmap)

        # Cell points
        points = Cell.center_point_array.active

        if parameter is None:
            # Case: No parameter → color all points black and hide colorbar
            self.scatter.set_color('black')     # Set uniform black color
            self.scatter.set_array(None)        # Disable the color
            self.cb.remove()                    # Remove unused colorbar
            self.scatter.set_offsets(points)    # Set the points

        else:
            # Add the coloring based on the parameter
            values = getattr(Cell, f"{parameter}_array").active

            if sort_values:
                # Sort from the lowest value to highest
                sort_indexes = np.argsort(values)
                points = points[sort_indexes]
                values = values[sort_indexes]

            self.scatter.set_offsets(points)    # Set the points
            self.scatter.set_array(values)      # Set coloring values
            colorbar_title = parameter.replace('_', ' ').title()
            self.cb.set_label(colorbar_title)   # Change label to the currently used parameter

            # Set new colorbar max
            param_max = np.max(values)
            print(f"param max {param_max}")
            self.scatter.set_clim(vmin=0, vmax=param_max)

        # Show the plot
        plt.show()

    def plot_cells_2D_colony(self, colony: Colony, parameter: str = None, cmap: str = None,
                      sort_values: bool = False):
        # Copy template
        self.fig, self.ax = plt.subplots(dpi=self.dpi)
        self.scatter = self.ax.scatter([], [], c=[], s=self.dotsize, cmap='Spectral', linewidths=0)
        self.cb = self.fig.colorbar(self.scatter, ax=self.ax)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')

        colony_min = np.min(Cell.center_point_array[colony.cell_indexes, :])
        colony_max = np.max(Cell.center_point_array[colony.cell_indexes, :])
        diameter = colony_max - colony_min
        border_buffer = 0.05 * diameter
        self.ax.set_xlim(colony_min - border_buffer, colony_max + border_buffer)
        self.ax.set_ylim(colony_min - border_buffer, colony_max + border_buffer)

        # Change color map
        if cmap is not None:
            self.scatter.set_cmap(cmap)

        # Cell points
        points = Cell.center_point_array.active

        if parameter is None:
            # Case: No parameter → color all points black and hide colorbar
            self.scatter.set_color('black')     # Set uniform black color
            self.scatter.set_array(None)        # Disable the color
            self.cb.remove()                    # Remove unused colorbar
            self.scatter.set_offsets(points)    # Set the points

        else:
            # Add the coloring based on the parameter
            values = getattr(Cell, f"{parameter}_array").active

            if sort_values:
                # Sort from the lowest value to highest
                sort_indexes = np.argsort(values)
                points = points[sort_indexes]
                values = values[sort_indexes]

            self.scatter.set_offsets(points)    # Set the points
            self.scatter.set_array(values)      # Set coloring values
            colorbar_title = parameter.replace('_', ' ').title()
            self.cb.set_label(colorbar_title)   # Change label to the currently used parameter

            # Set new colorbar max
            param_max = np.max(values)
            print(f"param max {param_max}")
            self.scatter.set_clim(vmin=0, vmax=param_max)

        # Show the plot
        plt.show()

    @staticmethod
    def plot_cells_3D_crowding_index():
        """[WIP]"""
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

import json
import numpy as np
from pathlib import Path
from src.utils.dynamic_array import DynamicArray, Dynamic2DArray
from src.algorithm.cell_based.cell import Cell
from src.algorithm.cell_based.colony import Colony


class CellDataManager:
    def __init__(self, data_path: Path):
        self.data_path: Path = data_path
        self.array_data_path: Path = data_path.with_name(data_path.stem + "_arrays").with_suffix(".npz")
        self.cell_data_path: Path = data_path.with_name(data_path.stem + "_cell_colony_data").with_suffix(".json")

    def save_array_data(self):
        np.savez_compressed(
            self.array_data_path,
            center=Cell.center_point_array.active,
            end=Cell.end_point_array.active,
            age=Cell.age_array.active,
            crowding=Cell.crowding_index_array.active,
            DivIVA=Cell.DivIVA_array.active
        )

    def save_cell_simulation_data(self):
        # Save arrays
        self.save_array_data()

        # Save cell class data
        cell_data: list[dict] = [
            {
                "index": cell.index,
                "parent": cell.parent.index if cell.parent else None,
                "children": [child.index for child in cell.children],
                "direction": cell.direction,
                "length": cell.length,
                "state_index": cell.state.index,
                "colony_index": cell.colony_index
            }
            for cell in Cell.instances
        ]

        # Save colony class data
        colony_data: list[dict] = [
            {
                "index": colony.index,
                "root_index": colony.root.index,
                "cell_indexes": list(colony.cell_indexes.active)
            }
            for colony in Colony.instances
        ]

        self.save_to_json(cell_data, colony_data)

    def save_to_json(self, cell_data, colony_data):
        def json_serializer(obj):
            """Handle non-JSON types automatically"""
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return vars(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable.")

        with open(self.cell_data_path, 'w') as jfile:
            json.dump(
                {
                    "cells": cell_data,
                    "colonies": colony_data
                },
                jfile, default=json_serializer
            )

    def load_all_simulation_data(self):
        # Reset the classes
        Cell.reset_class()
        Colony.reset_class()

        # Load in array data
        array_data = np.load(self.array_data_path)
        self.load_array_data(array_data)

        # Load in cell class data
        all_data = json.load(open(self.cell_data_path))
        cell_data = all_data["cells"]
        self.load_cell_data(cell_data)
        colony_data = all_data["colonies"]
        self.load_colony_data(colony_data)

        # Make sure that all cells are relinked to the correct colony
        self.relink_cells_to_colony()

    @staticmethod
    def load_array_data(array_data):
        Cell.center_point_array = Dynamic2DArray.load_data(array_data["center"])
        Cell.end_point_array = Dynamic2DArray.load_data(array_data["end"])
        Cell.age_array = DynamicArray.load_data(array_data["age"])
        Cell.crowding_index_array = DynamicArray.load_data(array_data["crowding"])
        Cell.DivIVA_array = DynamicArray.load_data(array_data["DivIVA"])

    @staticmethod
    def load_cell_data(cell_data):
        for data in cell_data:
            Cell.load_data(data)

    @staticmethod
    def load_colony_data(colony_data):
        for data in colony_data:
            Colony.load_data(data)

    @staticmethod
    def relink_cells_to_colony():
        for cell in Cell.instances:
            Colony.instances[cell.colony_index].add_cell(cell)



import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from pathlib import Path
import json
from src.utils.load_config import Config
from src.utils.instance_tracker import InstanceTracker
from src.utils.visual.colony_analysis_plotter import CAVisualize
from src.algorithm.cell_based.colony import Colony
from src.algorithm.cell_based.cell import Cell
from src.algorithm.event.event import Event


class ColonyAnalysisReport(CAVisualize, InstanceTracker):
    # Meta reporter params
    config_params = [
        "CELL_SEGMENT_LENGTH",
        "GROWTH_RATE",
        "CROWDING_SLOPE_STEEPNESS",
        "CROWDING_ALPHA",
        "TROPISM_ALPHA",
        "TROPISM_MAX_BEND"
    ]
    report_params = [
        "min_diameter",
        "max_diameter",
        "number_of_cells",
        "num_active_cells",
        "max_crowding",
        "average_crowding",
        "average_propensity",
        "area"
    ]
    metrics = [
        "log10_area",
        "total_length",
        "hyphal_density",
        "tip_density"
    ]

    def __init__(self, colonies: list[Colony]):
        super().__init__()
        self.colonies = colonies
        self.total_colonies: int = len(colonies)
        self.number_of_cells: list[int] = []
        self.num_active_cells: list[int] = []
        self.area: list[float] = []

        self.min_diameter: list[np.float64] = []
        self.max_diameter: list[np.float64] = []
        self.pca_projection = []

        self.average_propensity: list[np.float64] = []
        self.propensity_distr: list[np.ndarray] = []
        self.max_crowding: list[np.float64] = []
        self.average_crowding: list[np.float64] = []
        self.crowd_distr: list[np.ndarray] = []

        self.simulator = None

    def run_analysis(self):
        for colony in self.colonies:
            self.fill_data(colony)

    def fill_data(self, colony: Colony):
        # Cell stats
        self.number_of_cells.append(colony.cell_count)
        act_cells, props, crowding = self.cell_data_distribution(colony)
        self.num_active_cells.append(act_cells)
        self.average_propensity.append(np.average(props))
        self.propensity_distr.append(props)
        self.max_crowding.append(max(crowding))
        self.average_crowding.append(np.average(crowding))
        self.crowd_distr.append(crowding)

        # Morphology repeat_data
        points = colony.cell_points
        # area, min_dim, max_dim = self.calc_hull(points)
        self.area.append(self.calc_area(points))
        min_dim, max_dim, projected = self.calc_diameter(points)
        # self.area.append(area)
        self.min_diameter.append(min_dim)
        self.max_diameter.append(max_dim)
        self.pca_projection.append(projected)

    @staticmethod
    def calc_area(points: np.ndarray):
        if len(points) < 3:
            return 0

        hull = ConvexHull(points)
        area = hull.volume  # 2D volume = area
        return area

    @staticmethod
    def minimal_distance(hull):
        min_dist = np.inf
        for simplex in hull.simplices:
            # Get two points on the simplex (edge)
            p1, p2 = hull.points[simplex[0]], hull.points[simplex[1]]
            edge_length = np.linalg.norm(p1 - p2)
            if edge_length < min_dist:
                min_dist = edge_length
        return min_dist

    @staticmethod
    def maximum_distance(hull):
        points = hull.points[hull.vertices]
        pairwise_distances = cdist(points, points)
        return np.max(pairwise_distances)

    @staticmethod
    def calc_diameter(points: np.ndarray) -> (float, float, np.ndarray):
        if len(points) < 3:
            return 0, 0, points

        center = points.mean(axis=0)
        centered_points = points - center

        U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)
        components = Vt.T

        projected = centered_points @ components
        PC1 = projected[:, 0]
        PC2 = projected[:, 1]

        PC1_span = max(PC1) - min(PC1)
        PC2_span = max(PC2) - min(PC2)

        return min(PC1_span, PC2_span), max(PC1_span, PC2_span), projected

    @staticmethod
    def cell_data_distribution(colony: Colony) -> tuple[int, np.ndarray, np.ndarray]:
        cell_inds = colony.cell_indexes
        cell_propensities: np.ndarray = Event.event_propensities_array[cell_inds, :].sum(axis=1)
        cell_active_mask: np.ndarray = cell_propensities > 0
        cell_props_filt: np.ndarray = cell_propensities[cell_active_mask]
        active_cells: int = cell_props_filt.size
        crowding: np.ndarray = Cell.crowding_index_array[cell_inds]
        return active_cells, cell_props_filt, crowding

    def save_as_formatted_report(self, run_file_path, time_point):
        text_lines = [f"=[Report: {self.index}]=",
                      self.format_line("Time point", [time_point])]
        for param_name in self.report_params:
            param_values = getattr(self, param_name)
            text_lines.append(self.format_line(param_name, param_values))

        total_length = [value * Config().cell.CELL_SEGMENT_LENGTH for value in self.number_of_cells]
        text_lines.append(self.format_line('total_lengt', total_length))
        density = [
            length / area
            if area > 0
            else 0
            for length, area in zip(total_length, self.area)
        ]
        text_lines.append(self.format_line('density', density))
        single_string_report = "\n".join(text_lines) + "\n\n"
        with open(run_file_path, 'a') as file:
            file.write(single_string_report)

    def save_as_json(self, repeat_file_path: Path, time_point: float):
        # Calculated metrics parameters
        metric_parameters = self.get_metric_values()

        # Report repeat_data in json format
        parameters: dict = {
            param_name: getattr(self, param_name)
            for param_name in self.report_params
        }
        parameters.update(metric_parameters)

        report_json = {
            "Report_ID": self.index,
            "Time_Point": time_point,
            "Parameters": parameters
        }

        # Convert repeat_data to json datatypes
        def json_serializer(obj):
            """Handle non-JSON types automatically"""
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return vars(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable.")

        # Append repeat_data to the run file
        with open(repeat_file_path.with_suffix(".jsonl"), 'a') as jfile:
            json.dump(report_json, jfile, default=json_serializer)
            jfile.write("\n")    # New line on the end of each report (that's why jsonl instead of json)

    def get_metric_values(self):
        log10_area = [
                np.log10(area)
                if area > 0 else 1e-6
                for area in self.area
            ]
        total_length = [
            value * Config().cell.CELL_SEGMENT_LENGTH
            for value in self.number_of_cells
        ]
        hyphal_density = [
            length / area
            if area > 0 else 0
            for length, area in zip(total_length, self.area)
        ]
        tip_density = [
            active_tips / length
            for active_tips, length in zip(self.num_active_cells, total_length)
        ]
        complete_metrics = {
            "log10_area": log10_area,
            "total_length": total_length,
            "hyphal_density": hyphal_density,
            "tip_density": tip_density
        }
        return {
            metric_name: value
            for metric_name, value in complete_metrics.items()
            if metric_name in self.metrics
        }

    @staticmethod
    def to_string(data):
        return list(map(str, data))

    def format_line(self, param_name: str, data):
        return f"{param_name: <20}: {' '.join(self.to_string(data))}"

    def set_simulator(self, simulator):
        self.simulator = simulator
        return self

    def show_plots(self, extra_title: str = ''):
        for i in range(self.total_colonies):
            self.show_PCA(self.pca_projection, i)
            self.show_propensity(self.propensity_distr, i)
            self.show_crowding(self.crowd_distr, i, extra_title)

    # @classmethod
    # def meta_analysis(cls):
    #     def to_string(format_list):
    #         return list(map(str, format_list))
    #
    #     def format_config(param_name: str, config):
    #         format = str(getattr(config, param_name))
    #         print(f"{param_name: <20}: {format}")
    #
    #     def format_report(param_name: str, data: list, averages: dict[str, list]):
    #         if param_name not in averages:
    #             averages[param_name] = []
    #         averages[param_name].append(np.average(data))
    #         print(f"{param_name: <20}: {' '.join(to_string(data))}")
    #
    #     print("=[Config params]")
    #     for config_param in cls.config_params:
    #         format_config(config_param, CellConfig)
    #
    #     param_avr = {}
    #
    #     for report in cls.instances:
    #         report: ColonyAnalysisReport = report
    #         print(f"=[Report: {report.index}]")
    #         for param in cls.report_params:
    #             param_data = getattr(report, param)
    #             format_report(param, param_data, param_avr)
    #
    #         total_length = [value * CellConfig.CELL_SEGMENT_LENGTH for value in report.number_of_cells]
    #         format_report('total_lengt', total_length, param_avr)
    #         density = [length / area for length, area in zip(total_length, report.area)]
    #         format_report('density', density, param_avr)
    #         print("#############")
    #
    #     # print(f"{'[Averagesreport]': <20}: {' '.join(to_string(range(len(cls.instances))))}")
    #     print(f"{CellConfig.CELL_SEGMENT_LENGTH: <20}: {' '.join(to_string(range(len(cls.instances))))}")
    #     for param, averages in param_avr.items():
    #         print(f"{param: <20}: {' '.join(to_string(averages))}")


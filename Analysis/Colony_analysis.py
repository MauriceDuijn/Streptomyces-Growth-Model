import numpy as np
from scipy.spatial import ConvexHull

from utils.Instance_tracker import InstanceTracker
from utils.Colony_analysis_visualize import CAVisualize
from Cell_manager.Colony import Colony
from Cell_manager.Cell import Cell
from Event_manager.Event import Event


class ColonyAnalysis(CAVisualize, InstanceTracker):
    def __init__(self, colonies: list[Colony]):
        super().__init__()
        self.colonies = colonies
        self.total_colonies: int = len(colonies)
        self.sizes: list[int] = []
        self.area: list[float] = []

        self.min_diameter: list[np.float64] = []
        self.max_diameter: list[np.float64] = []
        self.pca_projection = []

        self.num_active_cells: list[int] = []
        self.propensity_distr: list[np.ndarray] = []
        self.crowd_distr: list[np.ndarray] = []

    def run_analysis(self):
        for colony in self.colonies:
            self.fill_data(colony)

    def fill_data(self, colony: Colony):
        self.sizes.append(colony.cell_count)

        points = colony.cell_points
        self.area.append(self.calc_area(points))
        min_dim, max_dim, projected = self.calc_min_max_diameter(points)
        self.min_diameter.append(min_dim)
        self.max_diameter.append(max_dim)
        self.pca_projection.append(projected)

        act_cells, props, crowding = self.cell_data_distribution(colony)
        self.num_active_cells.append(act_cells)
        self.propensity_distr.append(props)
        self.crowd_distr.append(crowding)

    @staticmethod
    def calc_area(points: np.ndarray):
        hull = ConvexHull(points)
        area = hull.volume  # 2D volume = area
        return area

    @staticmethod
    def calc_min_max_diameter(points: np.ndarray):
        center = points.mean(axis=0)
        centered_points = points - center

        U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)
        components = Vt.T

        projected = centered_points @ components
        PC1 = projected[:, 0]
        PC2 = projected[:, 1]

        PC1_length = max(PC1) - min(PC1)
        PC2_length = max(PC2) - min(PC2)

        return PC1_length, PC2_length, projected

    @staticmethod
    def cell_data_distribution(colony: Colony) -> tuple[int, np.ndarray, np.ndarray]:
        cell_inds = colony.cell_indexes
        cell_propensities: np.ndarray = Event.event_propensities_array[cell_inds, :].sum(axis=1)
        cell_active_mask: np.ndarray = cell_propensities > 0
        cell_props_filt: np.ndarray = cell_propensities[cell_active_mask]
        active_cells: int = cell_props_filt.size
        crowding: np.ndarray = Cell.crowding_index_array[cell_inds]
        return active_cells, cell_props_filt, crowding

    def show_plots(self):
        for i in range(self.total_colonies):
            self.show_PCA(self.pca_projection, i)
            self.show_propensity(self.propensity_distr, i)
            self.show_crowding(self.crowd_distr, i)

    @classmethod
    def meta_analysis(cls):
        def format_line(param_name: str, report: ColonyAnalysis):
            param_format = [str(value) for value in getattr(report, param_name)]
            print(f"{param_name: <20}: {' '.join(param_format)}")

        for report in cls.instances:
            report: ColonyAnalysis = report
            print(f"=[report]: {report.index}")
            format_line("sizes", report)
            format_line("area", report)
            format_line("min_diameter", report)
            format_line("max_diameter", report)
            format_line("num_active_cells", report)
            print("#############")

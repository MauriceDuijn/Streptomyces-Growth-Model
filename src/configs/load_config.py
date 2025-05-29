import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunConfig:
    RUN_NAME: str
    RUN_REPEATS: int
    END_TIME: float


@dataclass
class CellConfig:
    SPORE_AMOUNT: int                   # Amount of spores
    CELL_SEGMENT_LENGTH: float          # Segment size in µm
    GROWTH_SPEED: float                 # Micrometer growth per time unit
    ANGLE_DEVIATION_PER_MICRON: float   # Angle noise deviation (in degrees) per µm
    CROWDING_SLOPE_STEEPNESS: float     # Controls the crowding influence radius
    CROWDING_INTENSITY: float           # Strength of the crowding effect
    ERROR_TOLERANCE: float              # Cutoff value for weak crowding values
    TROPISM_INTENSITY: float            # Strength of stimulus difference
    TROPISM_BEND_PER_MICRON: float      # Maximum bend from stimulus per µm

    # Normalize parameters by segment length
    GROWTH_RATE: float = field(init=False)
    NOISE_ANGLE_DEVIATION: float = field(init=False)
    CROWDING_ALPHA: float = field(init=False)
    TROPISM_ALPHA: float = field(init=False)
    TROPISM_MAX_BEND: float = field(init=False)

    def normalize(self):
        self.GROWTH_RATE = self.GROWTH_SPEED / self.CELL_SEGMENT_LENGTH
        self.NOISE_ANGLE_DEVIATION = self.ANGLE_DEVIATION_PER_MICRON * self.CELL_SEGMENT_LENGTH ** 0.5
        self.CROWDING_ALPHA = self.CROWDING_INTENSITY
        self.TROPISM_ALPHA = self.TROPISM_INTENSITY
        self.TROPISM_MAX_BEND = self.TROPISM_BEND_PER_MICRON


@dataclass
class DivIVAConfig:
    DIVIVA_MAX_BINDINGRATE: float
    INITIAL_SPROUT_DIVIVA: float
    SPLIT_THRESHOLD: float
    SPLIT_SENSITIVITY: float
    SPLIT_RATIO: float
    BRANCH_SPROUT_THRESHOLD: float
    BRANCH_SPROUT_SENSITIVITY: float


@dataclass
class ChemicalConfig:
    INIT_STARCH_AMOUNT: int
    STARCH_RATE: float = field(init=False)

    def normalize(self, cell_config: CellConfig):
        self.INIT_STARCH_AMOUNT /= cell_config.CELL_SEGMENT_LENGTH
        self.STARCH_RATE = cell_config.GROWTH_RATE / self.INIT_STARCH_AMOUNT


@dataclass
class LoggerConfig:
    log_element_count: bool = True
    log_reaction_propensity: bool = True
    log_state_count: bool = True
    log_total_propensity: bool = True
    log_cell_total_propensity: bool = True


@dataclass
class ReporterConfig:
    SAVE_AS_JSON_FORMAT: bool
    REPORT_INTERVAL: float
    ACTIVE_PARAMETERS: tuple[str]
    ACTIVE_METRICS: tuple[str]
    TIME_POINTS: list[float] = field(init=False)
    POSSIBLE_PARAMETERS: tuple[str] = field(
        default=(
            "min_diameter",
            "max_diameter",
            "number_of_cells",
            "num_active_cells",
            "max_crowding",
            "average_crowding",
            "average_propensity",
            "area"
        )
    )
    POSSIBLE_METRICS: tuple[str] = field(
        default=(
            "log10_area",
            "total_length",
            "hyphal_density",
            "tip_density"
        )
    )

    def format_time_points(self, run_config: RunConfig):
        self.TIME_POINTS: list[float] = [
            i * self.REPORT_INTERVAL
            for i in range(int(run_config.END_TIME / self.REPORT_INTERVAL) + 1)
        ]


@dataclass
class PlotterConfig:
    DPI: int = 400
    DOT_SIZE: int = 5


class Config:
    """
    Central config class.
    Is a singleton, recalling the class will reference the first instance.
    """
    _instance: "Config" = None
    config_path: Path = Path("src/configs")

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_configs()
        return cls._instance

    def _init_configs(self):
        self.run: RunConfig = self._load_yaml("run_config.yaml", RunConfig)
        self.cell: CellConfig = self._load_yaml("cell_config.yaml", CellConfig)
        self.diviva: DivIVAConfig = self._load_yaml("DivIVA_config.yaml", DivIVAConfig)
        self.chem: ChemicalConfig = self._load_yaml("chemical_config.yaml", ChemicalConfig)
        self.log: LoggerConfig = LoggerConfig()
        self.report: ReporterConfig = self._load_yaml("report_config.yaml", ReporterConfig)
        self.plotter: PlotterConfig = PlotterConfig()

        self.cell.normalize()
        self.chem.normalize(self.cell)
        self.report.format_time_points(self.run)

    @classmethod
    def _load_yaml(cls, yaml_file_name, config_cls):
        yaml_path = cls.config_path / yaml_file_name
        with open(yaml_path) as file:
            data = yaml.safe_load(file)
        return config_cls(**data)

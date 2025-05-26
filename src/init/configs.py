from dataclasses import dataclass


@dataclass
class GlobalConfig:
    RUN_NAME: str = "H48_M20_1.0"
    RUN_REPEATS: int = 1
    REPORT_INTERVAL: float = 6
    END_TIME: float = 48
    SPORE_AMOUNT: int = 20
    ERROR_TOLERANCE_SIGNIFICANCE = 3


@dataclass
class CellConfig:
    CELL_SEGMENT_LENGTH: float = 1          # Segment size in micrometers
    GROWTH_SPEED: float = 10                # Micrometer growth per time unit
    ANGLE_DEVIATION_PER_MICRON: float = 20  # Normal noise deviation in degrees per micro meter
    CROWDING_SLOPE_STEEPNESS: float = 10    # Width factor for crowding steepness, regulates the radius of influence
    CROWDING_INTENSITY: float = 1e-2        # How intens the crowding index affects per micro meter of segment
    TROPISM_INTENSITY = 0                   # How intens the stimulus difference affects the bend
    TROPISM_BEND_PER_MICRON = 20            # Maximum bend caused by tropism

    # Normal
    GROWTH_RATE: float = GROWTH_SPEED / CELL_SEGMENT_LENGTH
    # NOISE_ANGLE_DEVIATION: float = ANGLE_DEVIATION_PER_MICRON * CELL_SEGMENT_LENGTH ** np.log(2)
    NOISE_ANGLE_DEVIATION: float = ANGLE_DEVIATION_PER_MICRON * CELL_SEGMENT_LENGTH ** 0.5
    CROWDING_ALPHA: float = CROWDING_INTENSITY

    TROPISM_ALPHA: float = TROPISM_INTENSITY
    TROPISM_MAX_BEND: float = TROPISM_BEND_PER_MICRON

    @classmethod
    def normalize_segment_length(cls):
        """Normalize parameters based on segment length"""
        cls.GROWTH_RATE = cls.GROWTH_SPEED / cls.CELL_SEGMENT_LENGTH
        cls.NOISE_ANGLE_DEVIATION = cls.ANGLE_DEVIATION_PER_MICRON * cls.CELL_SEGMENT_LENGTH ** 0.5


@dataclass
class DivIVAConfig:
    DIVIVA_MAX_BINDINGRATE: float = 1e-1
    INITIAL_SPROUT_DIVIVA: float = 1
    SPLIT_THRESHOLD: float = 1.1
    SPLIT_SENSITIVITY: float = 1
    SPLIT_RATIO: float = 0.8
    BRANCH_SPROUT_THRESHOLD: float = 1
    BRANCH_SPROUT_SENSITIVITY: float = 1


@dataclass
class ChemicalConfig:
    INIT_STARCH_AMOUNT: int = 1e6 / CellConfig.CELL_SEGMENT_LENGTH
    STARCH_RATE: float = CellConfig.GROWTH_RATE / INIT_STARCH_AMOUNT  # Max cell event propensity = growth speed

    @classmethod
    def update_starch_rate(cls):
        cls.STARCH_RATE = CellConfig.GROWTH_RATE / cls.INIT_STARCH_AMOUNT


@dataclass
class LoggerConfig:
    log_element_count: bool = True
    log_reaction_propensity: bool = True
    log_state_count: bool = True
    log_total_propensity: bool = True
    log_cell_total_propensity: bool = True


class ReporterConfig:
    SAVE_AS_JSON_FORMAT = True
    TIME_POINTS: list[float] = [
        i * GlobalConfig.REPORT_INTERVAL
        for i in range(int(GlobalConfig.END_TIME / GlobalConfig.REPORT_INTERVAL) + 1)
    ]
    POSSIBLE_PARAMETERS = [
        "min_diameter",
        "max_diameter",
        "number_of_cells",
        "num_active_cells",
        "average_crowding",
        "average_propensity",
        "area"
    ]
    ACTIVE_PARAMETERS = POSSIBLE_PARAMETERS


@dataclass
class PlotterConfig:
    DPI: int = 400
    DOT_SIZE: int = 5
    LINE_WIDTH: int = 0


@dataclass
class AnimatorConfig:
    DPI: int = 200
    FPS: int = 5
    DOT_SIZE: int = 3
    PARAMETER: str = "crowding_index"

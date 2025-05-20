from dataclasses import dataclass


@dataclass
class GlobalConfig:
    END_TIME: float = 60
    SPORE_AMOUNT: int = 1
    ERROR_TOLERANCE_SIGNIFICANCE = 3


@dataclass
class CellConfig:
    CELL_SEGMENT_LENGTH: float = 1          # Segment size in micrometers
    GROWTH_SPEED: float = 10                # Micrometer growth per time unit
    NOISE_ANGLE_DEVIATION: float = 20       # Normal noise deviation in degrees per micro meter
    CROWDING_SLOPE_STEEPNESS: float = 10    # Width factor for crowding steepness
    CROWDING_ALPHA: float = 0 #5e-3            # How intens the crowding index affects the a micro meter of segment
    TROPISM_SENSITIVITY = 0 #1e-2              # How intens the stimulus difference affects the bend
    TROPISM_MAX_BEND = 90                   # Maximum bend caused by tropism

    GROWTH_RATE: float = GROWTH_SPEED / CELL_SEGMENT_LENGTH

    @classmethod
    def normalize_segment_length(cls):
        """Normalize parameters based on segment length"""
        cls.GROWTH_RATE = cls.GROWTH_SPEED / cls.CELL_SEGMENT_LENGTH


@dataclass
class DivIVAConfig:
    DIVIVA_MAX_BINDINGRATE: float = 1e-1
    INITIAL_SPROUT_DIVIVA: float = 1
    SPLIT_THRESHOLD: float = 1.1
    SPLIT_RATE: float = 10
    SPLIT_RATIO: float = 0.8
    BRANCH_SPROUT_THRESHOLD = 1
    BRANCH_SPROUT_RATE = 1


@dataclass
class ChemicalConfig:
    INIT_STARCH_AMOUNT: int = 100_000
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

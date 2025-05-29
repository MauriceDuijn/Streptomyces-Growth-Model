import numpy as np
from types import SimpleNamespace

from src.configs.load_config import Config

from src.utils.analysis.simulation_logger import SimulationLogger
from src.utils.analysis.report_manager import ReportManager
from src.algorithm.chemistry.element import Element
from src.algorithm.chemistry.reaction import Reaction
from src.algorithm.event.state import State
from src.algorithm.event.condition import Condition
from src.algorithm.event.event import Event
from src.algorithm.cell_based.cell import Cell
from src.algorithm.cell_based import cell_action as ce_ac
from src.algorithm.cell_based.colony import Colony
from src.algorithm.spatial.spatial_hashing import SpatialHashing
from src.utils.visual.colony_structure_plotter import ColonyPlotter
from src.algorithm.gillespie_algorithm import GillespieSimulator


# ---------------
# Main simulation initialize
# ---------------
def init():
    # Create initial classes based on the configs
    reset_classes()     # In case there is some data remaining from a previouse repeat
    config = Config()   # Initialize central config class
    elements, reactions, states, conditions, general_actions, event_actions, events = create_classes(config)

    # Set the global partition size for the Colony SpatialHashing
    set_partition_size(config.cell.ERROR_TOLERANCE)

    # Create spores
    initialize_spores(states)

    # Create utils
    logger, reporter = create_analysis_tools(config)
    plotter, animator = create_utils(config)

    # Create simulation
    simulation = GillespieSimulator(config.run.END_TIME,
                                    Reaction.instances,
                                    Condition.instances,
                                    Event.instances,
                                    Cell.instances,
                                    logger=logger,
                                    reporter=reporter,
                                    animator=animator
                                    )

    return simulation


def reset_classes():
    for project_class in [Element, Reaction, State, Condition, Event, Cell, Colony]:
        project_class.reset_class()


def create_analysis_tools(config: Config):
    # Create logger
    logger: SimulationLogger = SimulationLogger(config.log)

    # Create reporter
    reporter: ReportManager = ReportManager(config.report)

    return logger, reporter


def create_utils(config: Config):
    # Create plotter
    ColonyPlotter.dpi = config.plotter.DPI
    plotter = ColonyPlotter(dot_size=config.plotter.DOT_SIZE)

    # Create animator
    # animator = CellGrowthAnimator(cells, AnimatorConfig.PARAMETER, GlobalConfig.END_TIME,
    #                               fps=AnimatorConfig.FPS, dot_size=1)
    animator = None

    return plotter, animator


def create_classes(config):
    # ---------------
    # Define Elements
    # ---------------
    elements = SimpleNamespace(
        STARCH=Element("Starch", 'S', config.chem.INIT_STARCH_AMOUNT)
    )

    # ---------------
    # Define Reactions
    # ---------------
    reactions = SimpleNamespace(
        CONSUME=Reaction("Consume nutriënts", config.chem.STARCH_RATE,
                         {elements.STARCH: 1}, {})
    )

    # ---------------
    # Define States
    # ---------------
    states = SimpleNamespace(
        SPORE_GERM_TUBE_1=State("Root germ tube 1"),
        SPORE_GERM_TUBE_2=State("Root germ tube 2"),
        STRAIGHT=State("Straight"),
        LATERAL=State("Lateral"),
        DORMANT=State("Dormant")
    )

    # ---------------
    # Define Conditions
    # ---------------
    conditions = SimpleNamespace(
        CROWDING_INDEX=Condition("Crowding index", "static", "crowding_index"),
        SPLIT_FOCI=Condition("split DivIVA threshold",
                             method_name="linear", parameter="DivIVA",
                             threshold=config.diviva.SPLIT_THRESHOLD, alpha=config.diviva.SPLIT_SENSITIVITY),
        BRANCH_SPROUT=Condition("sprout branch",
                                method_name="linear", parameter="DivIVA",
                                threshold=config.diviva.BRANCH_SPROUT_THRESHOLD, alpha=config.diviva.BRANCH_SPROUT_SENSITIVITY)
    )

    # ---------------
    # Link repeat_data from other classes to the action class
    # ---------------
    ce_ac.Action.event_propensities = Event.event_propensities_array
    ce_ac.Action.state_mask = State.cell_mask_array
    ce_ac.Action.condition_factors = Condition.cell_condition_factor_array

    ce_ac.CrowdingIndex.crowding_steepness = config.cell.CROWDING_SLOPE_STEEPNESS
    ce_ac.CrowdingIndex.spacing = config.cell.CELL_SEGMENT_LENGTH

    ce_ac.GrowCell.cell_length = config.cell.CELL_SEGMENT_LENGTH
    ce_ac.GrowCell.angle_deviation = config.cell.NOISE_ANGLE_DEVIATION
    ce_ac.GrowCell.tropism_sensitivity = config.cell.TROPISM_ALPHA
    ce_ac.GrowCell.tropism_max_bend = config.cell.TROPISM_MAX_BEND

    # ---------------
    # Define General Actions
    # ---------------
    general_actions = SimpleNamespace(
        SWITCH_STRAIGHT=ce_ac.SwitchState(states.STRAIGHT),
        GERMTUBE_DIVIVA=ce_ac.AddDivIVA(config.diviva.INITIAL_SPROUT_DIVIVA),
        TRANSFER_DIVIVA_ALL=ce_ac.Transfer("DivIVA", 1),
        TRANSFER_DIVIVA_SPLIT=ce_ac.Transfer("DivIVA", config.diviva.SPLIT_RATIO),
        CROWDING_INDEX=ce_ac.CrowdingIndex(conditions.CROWDING_INDEX, config.cell.CROWDING_ALPHA),
        FRAGMENT=ce_ac.Fragment(stump_state=states.SPORE_GERM_TUBE_2)
    )

    # ---------------
    # Define Event Actions
    # ---------------
    event_actions = SimpleNamespace(
        SPROUT_GERMTUBE_1=ce_ac.GrowCell(
            parent_cell_actions=[],
            new_cell_actions=[
                general_actions.SWITCH_STRAIGHT,
                general_actions.GERMTUBE_DIVIVA,
                general_actions.CROWDING_INDEX
            ],
            dou_actions=[]
        ),
        SPROUT_GERMTUBE_2=ce_ac.GrowCell(
            parent_cell_actions=[],
            new_cell_actions=[
                general_actions.SWITCH_STRAIGHT,
                general_actions.GERMTUBE_DIVIVA,
                general_actions.CROWDING_INDEX
            ],
            dou_actions=[],
            bend=180
        ),
        GROW_STRAIGHT=ce_ac.GrowCell(
            parent_cell_actions=[],
            new_cell_actions=[
                general_actions.SWITCH_STRAIGHT,
                general_actions.CROWDING_INDEX
            ],
            dou_actions=[general_actions.TRANSFER_DIVIVA_ALL],
        ),
        GROW_STRAIGHT_SPLIT=ce_ac.GrowCell(
            parent_cell_actions=[],
            new_cell_actions=[
                general_actions.SWITCH_STRAIGHT,
                general_actions.CROWDING_INDEX
            ],
            dou_actions=[general_actions.TRANSFER_DIVIVA_SPLIT]
        ),
        GROW_LATERAL=ce_ac.GrowCell(
            parent_cell_actions=[],
            new_cell_actions=[
                general_actions.SWITCH_STRAIGHT,
                general_actions.CROWDING_INDEX
            ],
            dou_actions=[general_actions.TRANSFER_DIVIVA_ALL],
            bend=90
        )
    )

    # ---------------
    # Define Events
    # ---------------
    events = SimpleNamespace(
        SPROUT_GERMTUBE_1=Event(
            "Grow first germ tube",
            ingoing_states=[states.SPORE_GERM_TUBE_1],
            outgoing_state=states.SPORE_GERM_TUBE_2,
            conditions=[],
            action=event_actions.SPROUT_GERMTUBE_1,
            chemical_channel=reactions.CONSUME
        ),
        SPROUT_GERMTUBE_2=Event(
            "Grow second germ tube",
            ingoing_states=[states.SPORE_GERM_TUBE_2],
            outgoing_state=states.DORMANT,
            conditions=[],
            action=event_actions.SPROUT_GERMTUBE_2,
            chemical_channel=reactions.CONSUME
        ),
        GROW_TIP=Event(
            "Grow normal cell from the tip",
            ingoing_states=[states.STRAIGHT],
            outgoing_state=states.DORMANT,
            conditions=[conditions.CROWDING_INDEX],
            action=event_actions.GROW_STRAIGHT,
            chemical_channel=reactions.CONSUME
        ),
        GROW_TIP_SPLIT=Event(
            "Grow cell from the tip and split DivIVA foci",
            ingoing_states=[states.STRAIGHT],
            outgoing_state=states.LATERAL,
            conditions=[conditions.CROWDING_INDEX, conditions.SPLIT_FOCI],
            action=event_actions.GROW_STRAIGHT_SPLIT,
            chemical_channel=reactions.CONSUME
        ),
        GROW_LATERAL=Event(
            "Grow a lateral branch from split DivIVA foci",
            ingoing_states=[states.LATERAL],
            outgoing_state=states.DORMANT,
            conditions=[conditions.CROWDING_INDEX, conditions.BRANCH_SPROUT],
            action=event_actions.GROW_LATERAL,
            chemical_channel=reactions.CONSUME
        ),
        # FRAGMENT=Event(
        #     "Fragment a branch into a new colony",
        #     ingoing_states=[states.DORMANT],
        #     outgoing_state=states.DORMANT,
        #     conditions=[],
        #     action=general_actions.FRAGMENT,
        #     chemical_channel=reactions.CONSUME
        # )
    )

    # ---------------
    # Cell initialize
    # ---------------
    Cell.DivIVA_binding_rate = config.diviva.DIVIVA_MAX_BINDINGRATE

    return elements, reactions, states, conditions, general_actions, event_actions, events


def set_partition_size(error_tolerance):
    partition_size = ce_ac.CrowdingIndex.calculate_query_size(error_tolerance)
    SpatialHashing.partition_size = partition_size
    print("[Partition size]", partition_size)


def initialize_spores(states):
    for i in range(Config().cell.SPORE_AMOUNT):
        # Create spore cell
        origin = (0, 0)
        random_direction = np.random.uniform(0, 360)
        new_root = Cell.create_root_cell(origin, random_direction,
                                         states.SPORE_GERM_TUBE_1)

        # Add to a unique colony
        Colony(new_root)

        # Add base values for the root cell
        State.cell_mask_array.append(new_root.state.event_mask)
        Condition.cell_condition_factor_array.append(0)
        Event.event_propensities_array.append(0)

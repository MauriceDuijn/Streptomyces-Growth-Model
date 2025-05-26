import numpy as np
from types import SimpleNamespace
from src.init.configs import GlobalConfig, CellConfig, DivIVAConfig, ChemicalConfig, LoggerConfig, ReporterConfig, PlotterConfig
from src.utils.Analysis.SimulationLogger import SimulationLogger
from src.utils.Analysis.ReportManager import ReportManager
from src.chemistry.element import Element
from src.chemistry.reaction import Reaction
from src.event.state import State
from src.event.condition import Condition
from src.event.event import Event
from src.organic.cell import Cell
from src.organic.colony import Colony
from src.organic import cell_action as CeAc
from src.spatial.SpatialHashing import SpatialHashing
from src.utils.colony_structure_plotter import ColonyPlotter
from src.algorithm.gillespie_algorithm import GillespieSimulator


# ---------------
# Main simulation initialize
# ---------------
def init(paremeter_change=None):
    reset_classes()

    if paremeter_change:
        change_config(paremeter_change)

    # Create initial classes based on the configs
    elements, reactions, states, conditions, general_actions, event_actions, events = create_classes()

    # Set the global partition size for the Colony SpatialHashing
    partition_size = CeAc.CrowdingIndex.calculate_query_size(GlobalConfig.ERROR_TOLERANCE_SIGNIFICANCE)
    SpatialHashing.partition_size = partition_size
    print("[Partition size]", partition_size)

    initialize_spores(states)

    # Create utils
    logger, reporter = create_analysis_tools()
    plotter, animator = create_utils()

    # Create simulation
    simulation = GillespieSimulator(GlobalConfig.END_TIME,
                                    Reaction.instances,
                                    Condition.condition_collection,
                                    Event.event_instances,
                                    Cell.instances,
                                    logger=logger,
                                    reporter=reporter,
                                    animator=animator
                                    )

    return plotter, animator, simulation


def change_config(paremeter_change):
    CellConfig.CELL_SEGMENT_LENGTH = paremeter_change
    CellConfig.normalize_segment_length()
    ChemicalConfig.update_starch_rate()
    print(f"[Parameter change {CellConfig.CELL_SEGMENT_LENGTH}]")
    print(f"[effect: GS={CellConfig.GROWTH_SPEED}, GR={CellConfig.GROWTH_RATE}, AD={CellConfig.NOISE_ANGLE_DEVIATION}, SR={ChemicalConfig.STARCH_RATE}]")


def reset_classes():
    for project_class in [Element, Reaction, State, Condition, Event, Cell, Colony]:
        project_class.reset_class()


def create_analysis_tools():
    # Create logger
    logger = SimulationLogger(LoggerConfig)

    # Create reporter
    reporter = ReportManager(time_points=ReporterConfig.TIME_POINTS,
                             report_params=ReporterConfig.ACTIVE_PARAMETERS,
                             save_as_json=ReporterConfig.SAVE_AS_JSON_FORMAT)

    return logger, reporter


def create_utils():
    # Create plotter
    ColonyPlotter.dpi = PlotterConfig.DPI
    plotter = ColonyPlotter(dot_size=PlotterConfig.DOT_SIZE)

    # Create animator
    # animator = CellGrowthAnimator(cells, AnimatorConfig.PARAMETER, GlobalConfig.END_TIME,
    #                               fps=AnimatorConfig.FPS, dot_size=1)
    animator = None

    return plotter, animator


def create_classes():
    # ---------------
    # Define Elements
    # ---------------
    elements = SimpleNamespace(
        STARCH=Element("Starch", 'S', ChemicalConfig.INIT_STARCH_AMOUNT)
    )

    # ---------------
    # Define Reactions
    # ---------------
    reactions = SimpleNamespace(
        CONSUME=Reaction("Consume nutriënts", ChemicalConfig.STARCH_RATE,
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
                             threshold=DivIVAConfig.SPLIT_THRESHOLD, alpha=DivIVAConfig.SPLIT_SENSITIVITY),
        BRANCH_SPROUT=Condition("sprout branch",
                                method_name="linear", parameter="DivIVA",
                                threshold=DivIVAConfig.BRANCH_SPROUT_THRESHOLD, alpha=DivIVAConfig.BRANCH_SPROUT_SENSITIVITY)
    )

    # ---------------
    # Link repeat_data from other classes to the action class
    # ---------------
    CeAc.Action.event_propensities = Event.event_propensities_array
    CeAc.Action.state_mask = State.cell_mask_array
    CeAc.Action.condition_factors = Condition.cell_condition_factor_array

    CeAc.CrowdingIndex.k = CellConfig.CROWDING_SLOPE_STEEPNESS
    CeAc.CrowdingIndex.spacing = CellConfig.CELL_SEGMENT_LENGTH

    CeAc.GrowCell.cell_length = CellConfig.CELL_SEGMENT_LENGTH
    CeAc.GrowCell.angle_deviation = CellConfig.NOISE_ANGLE_DEVIATION
    CeAc.GrowCell.tropism_sensitivity = CellConfig.TROPISM_ALPHA
    CeAc.GrowCell.tropism_max_bend = CellConfig.TROPISM_MAX_BEND

    # ---------------
    # Define General Actions
    # ---------------
    general_actions = SimpleNamespace(
        SWITCH_STRAIGHT=CeAc.SwitchState(states.STRAIGHT),
        GERMTUBE_DIVIVA=CeAc.AddDivIVA(DivIVAConfig.INITIAL_SPROUT_DIVIVA),
        TRANSFER_DIVIVA_ALL=CeAc.Transfer("DivIVA", 1),
        TRANSFER_DIVIVA_SPLIT=CeAc.Transfer("DivIVA", DivIVAConfig.SPLIT_RATIO),
        CROWDING_INDEX=CeAc.CrowdingIndex(conditions.CROWDING_INDEX, CellConfig.CROWDING_ALPHA),
        FRAGMENT=CeAc.Fragment(stump_state=states.SPORE_GERM_TUBE_2)
    )

    # ---------------
    # Define Event Actions
    # ---------------
    event_actions = SimpleNamespace(
        SPROUT_GERMTUBE_1=CeAc.GrowCell(
            parent_cell_actions=[],
            new_cell_actions=[
                general_actions.SWITCH_STRAIGHT,
                general_actions.GERMTUBE_DIVIVA,
                general_actions.CROWDING_INDEX
            ],
            dou_actions=[]
        ),
        SPROUT_GERMTUBE_2=CeAc.GrowCell(
            parent_cell_actions=[],
            new_cell_actions=[
                general_actions.SWITCH_STRAIGHT,
                general_actions.GERMTUBE_DIVIVA,
                general_actions.CROWDING_INDEX
            ],
            dou_actions=[],
            bend=180
        ),
        GROW_STRAIGHT=CeAc.GrowCell(
            parent_cell_actions=[],
            new_cell_actions=[
                general_actions.SWITCH_STRAIGHT,
                general_actions.CROWDING_INDEX
            ],
            dou_actions=[general_actions.TRANSFER_DIVIVA_ALL],
        ),
        GROW_STRAIGHT_SPLIT=CeAc.GrowCell(
            parent_cell_actions=[],
            new_cell_actions=[
                general_actions.SWITCH_STRAIGHT,
                general_actions.CROWDING_INDEX
            ],
            dou_actions=[general_actions.TRANSFER_DIVIVA_SPLIT]
        ),
        GROW_LATERAL=CeAc.GrowCell(
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
    Cell.DivIVA_binding_rate = DivIVAConfig.DIVIVA_MAX_BINDINGRATE

    return elements, reactions, states, conditions, general_actions, event_actions, events


def initialize_spores(states):
    for i in range(GlobalConfig.SPORE_AMOUNT):
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

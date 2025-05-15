import numpy as np
from dataclasses import dataclass
from types import SimpleNamespace
from Chemistry_manager.ElementalSpecies import Element
from Chemistry_manager.ReactionChannel import Reaction
from Event_manager.State import State
from Event_manager.Condition import Condition
from Event_manager.Event import Event
from Cell_manager.Cell import Cell
from Cell_manager.Colony import Colony
import Cell_manager.CellAction as CeAc
from Space_manager.SpatialHashing import SpatialHashing
from utils.Plotter import Plotter
from utils.Animator import CellGrowthAnimator
from Event_manager.Gillespie_algorithm import GillespieSimulator


# ---------------
# Define Configs
# ---------------
@dataclass
class GlobalConfig:
    END_TIME: float = 48
    SPORE_AMOUNT: int = 1
    ERROR_TOLERANCE_SIGNIFICANCE = 3


@dataclass
class CellConfig:
    CELL_SEGMENT_LENGTH: float = 1
    GROWTH_SPEED: float = 10
    NOISE_ANGLE_DEVIATION: float = 10
    CROWDING_SLOPE_STEEPNESS: float = 10
    CROWDING_ALPHA: float = 1e-1
    TROPISM_SENSITIVITY = 0  # 2.5e-2
    TROPISM_MAX_BEND = 45


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
    STARCH_RATE: float = CellConfig.GROWTH_SPEED / INIT_STARCH_AMOUNT  # Max cell event propensity = growth speed


@dataclass
class PlotterConfig:
    DPI = 400
    DOT_SIZE = 100


@dataclass
class AnimatorConfig:
    DPI: int = 200
    FPS: int = 5
    DOT_SIZE: int = 3
    PARAMETER = "crowding_index"


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
                         threshold=DivIVAConfig.SPLIT_THRESHOLD, alpha=DivIVAConfig.SPLIT_RATE),
    BRANCH_SPROUT=Condition("sprout branch",
                            method_name="linear", parameter="DivIVA",
                            threshold=DivIVAConfig.BRANCH_SPROUT_THRESHOLD, alpha=DivIVAConfig.BRANCH_SPROUT_RATE)
)

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
# Link data from other classes to the action class
# ---------------
CeAc.Action.event_propensities = Event.event_propensities_array
CeAc.Action.state_mask = State.cell_mask_array
CeAc.Action.condition_factors = Condition.cell_condition_factor_array

CeAc.CrowdingIndex.k = CellConfig.CROWDING_SLOPE_STEEPNESS

CeAc.GrowCell.cell_length = CellConfig.CELL_SEGMENT_LENGTH
CeAc.GrowCell.angle_deviation = CellConfig.NOISE_ANGLE_DEVIATION
CeAc.GrowCell.tropism_sensitivity = CellConfig.TROPISM_SENSITIVITY
CeAc.GrowCell.tropism_max_bend = CellConfig.TROPISM_MAX_BEND

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


def initialize_spores():
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


# ---------------
# Main simulation initialize
# ---------------
def init():
    # Set the global partition size for the Colony SpatialHashing
    partition_size = CeAc.CrowdingIndex.calculate_query_size(GlobalConfig.ERROR_TOLERANCE_SIGNIFICANCE)
    print("[Partition size]", partition_size)
    SpatialHashing.partition_size = partition_size

    initialize_spores()

    # Create plotter
    Plotter.dpi = PlotterConfig.DPI
    plotter = Plotter(dot_size=PlotterConfig.DOT_SIZE)

    # Create animator
    # animator = CellGrowthAnimator(cells, AnimatorConfig.PARAMETER, GlobalConfig.END_TIME,
    #                               fps=AnimatorConfig.FPS, dot_size=1)
    animator = None

    # Create simulation
    simulation = GillespieSimulator(GlobalConfig.END_TIME,
                                    Reaction.reaction_channels,
                                    Condition.condition_collection,
                                    Event.event_instances,
                                    Cell.cell_collection,
                                    animator)

    return plotter, animator, simulation

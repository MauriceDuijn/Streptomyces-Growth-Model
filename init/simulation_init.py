import numpy as np
from Chemistry_manager.ElementalSpecies import Element
from Chemistry_manager.ReactionChannel import Reaction
from Event_manager.State import State
from Event_manager.Condition import Condition
from Event_manager.Event import Event
from Cell_manager.Cell import Cell
from Cell_manager.Colony import Colony
import Cell_manager.CellAction as act
from Space_manager.SpatialHashing import SpatialHashing
from Space_manager.SpatialPartitioning import SpacePartition
from utils.Plotter import Plotter
from utils.Animator import CellGrowthAnimator
from Event_manager.Gillespie_algorithm import GillespieSimulator


def init():
    # Global settings
    END_TIME = 1
    CROWDING_SLOPE_STEEPNESS = 10
    INFLUENCE_ALPHA = 2e-2
    ERROR_TOLERANCE_SIGNIFICANCE = 3

    # Cell settings
    INIT_CELL_AMOUNT = 1
    INIT_CELL_LENGTH = 1
    DIVIVA_BINDING_RATE = 1e-1
    UNIVERSAL_ANGLE_DEVIATION = 10
    GROWTH_SPEED = 10
    TROPISM_SENSITIVITY = 2.5e-2
    TROPISM_MAX_BEND = 45

    # Split settings
    SPLIT_THRESHOLD = 1.05
    SPLIT_RATE = 1
    SPLIT_RATIO = 0.8

    # Branch settings
    BRANCH_SPROUT_THRESHOLD = 1
    BRANCH_SPROUT_RATE = 1

    # Average branch-to-branch distance
    INITIAL_AMOUNT = SPLIT_THRESHOLD * (1 - SPLIT_RATIO)
    branch_to_branch_distance = GROWTH_SPEED / DIVIVA_BINDING_RATE * np.log(SPLIT_THRESHOLD / INITIAL_AMOUNT)
    print("[Branch-to-branch dist]", branch_to_branch_distance)

    # Chemical settings
    INIT_STARCH_AMOUNT = 100_000
    STARCH_RATE = GROWTH_SPEED / INIT_STARCH_AMOUNT  # Max cell event propensity = growth speed

    # Define elements
    elements = [
        Element("Starch", 'S', INIT_STARCH_AMOUNT)
    ]

    # Define reactions
    reactions = [
        Reaction("Consume starch", STARCH_RATE, {elements[0]: 1}, {})
    ]

    # Define states
    states = [
        State("Root 2 germ"),
        State("Root 1 germ"),
        State("Straight"),
        State("Lateral"),
        State("Dormant")
    ]

    # Define conditions
    conditions = [
        Condition("neighbour influence", "static", "crowding_index"),
        Condition("split DivIVA threshold", "linear", "DivIVA", threshold=SPLIT_THRESHOLD, alpha=SPLIT_RATE),
        Condition("sprout branch", "linear", "DivIVA", threshold=BRANCH_SPROUT_THRESHOLD, alpha=BRANCH_SPROUT_RATE)
    ]

    # Define general actions
    switch_straight = act.SwitchState(states[2])
    germ_cells_init_DivIVA = act.AddDivIVA(1)
    transfer_all_DivIVA = act.Transfer("DivIVA")
    transfer_split_DivIVA = act.Transfer("DivIVA", SPLIT_RATIO)
    crowding_index = act.CrowdingIndex(0, INFLUENCE_ALPHA)
    act.CrowdingIndex.k = CROWDING_SLOPE_STEEPNESS

    # Define event specific actions
    grow_germ_tube_1 = act.GrowCell([],
                                    [
                                        switch_straight,
                                        germ_cells_init_DivIVA,
                                        crowding_index
                                    ],
                                    [],
                                    cell_length=INIT_CELL_LENGTH,
                                    angle_deviation=UNIVERSAL_ANGLE_DEVIATION
                                    )

    grow_germ_tube_2 = act.GrowCell([],
                                    [
                                        switch_straight,
                                        germ_cells_init_DivIVA,
                                        crowding_index
                                     ],
                                    [],
                                    cell_length=INIT_CELL_LENGTH,
                                    angle_deviation=UNIVERSAL_ANGLE_DEVIATION,
                                    bend=180
                                    )

    grow_straight = act.GrowCell([],
                                 [
                                     switch_straight,
                                     crowding_index
                                 ],
                                 [transfer_all_DivIVA],
                                 cell_length=INIT_CELL_LENGTH,
                                 angle_deviation=UNIVERSAL_ANGLE_DEVIATION,
                                 tropism_sensitivity=TROPISM_SENSITIVITY,
                                 tropism_max_bend=TROPISM_MAX_BEND
                                 )

    grow_straight_split = act.GrowCell([],
                                       [
                                           switch_straight,
                                           crowding_index
                                       ],
                                       [transfer_split_DivIVA],
                                       cell_length=INIT_CELL_LENGTH,
                                       angle_deviation=UNIVERSAL_ANGLE_DEVIATION,
                                       tropism_sensitivity=TROPISM_SENSITIVITY,
                                       tropism_max_bend=TROPISM_MAX_BEND
                                       )

    grow_lateral = act.GrowCell([],
                                [
                                    switch_straight,
                                    crowding_index
                                ],
                                [transfer_all_DivIVA],
                                cell_length=INIT_CELL_LENGTH,
                                angle_deviation=UNIVERSAL_ANGLE_DEVIATION,
                                bend=90
                                )

    # Define events
    events = [
        Event("Grow first germ tube", [states[0]], states[1], [], grow_germ_tube_1, reactions[0]),
        Event("Grow second germ tube", [states[1]], states[-1], [], grow_germ_tube_2, reactions[0]),
        Event("Grow normal tip cell", [states[2]], states[-1], [0], grow_straight, reactions[0]),
        Event("Grow tip cell split DivIVA", [states[2]], states[3], [0, 1], grow_straight_split, reactions[0]),
        Event("Grow lateral", [states[3]], states[-1], [0, 2], grow_lateral, reactions[0]),
    ]

    # Define space partition size
    query_size = act.CrowdingIndex.calculate_query_size(ERROR_TOLERANCE_SIGNIFICANCE)
    print("[Query size]", query_size)
    SpatialHashing.partition_size = query_size

    # Link action parameters
    act.Action.event_propensities = Event.event_propensities_array
    act.Action.state_mask = State.cell_mask_array
    act.Action.condition_factors = Condition.cell_condition_factor_array

    # Define cell settings
    Cell.DivIVA_binding_rate = DIVIVA_BINDING_RATE
    for i in range(INIT_CELL_AMOUNT):
        # Random cell position
        # random_point = space.get_random_point()
        # random_point = (space.size / 2, space.size / 2)
        random_point = (0, 0)

        # Random cell direction
        # random_direction = 0
        random_direction = np.random.uniform(0, 360)

        # Create spore cell
        new_cell = Cell(random_point, random_point, random_direction, state=states[0], length=0)

        # Add to a unique colony
        new_colony = Colony()
        new_colony.add_cell(new_cell)

        # Add base values for the cell
        State.cell_mask_array.append(new_cell.state.event_mask)
        Condition.cell_condition_factor_array.append(0)
        Event.event_propensities_array.append(0)

    cells = Cell.cell_collection

    # Create plotter
    Plotter.dpi = 400
    plotter = Plotter(dot_size=1)

    # Create animator
    # animator = CellGrowthAnimator(space, cells, "crowding_index", END_TIME, fps=5, dot_size=1)
    animator = None

    # Create simulation
    simulation = GillespieSimulator(END_TIME,
                                    reactions,
                                    conditions,
                                    events,
                                    cells,
                                    animator)

    return END_TIME, elements, reactions, states, conditions, events, cells, plotter, animator, simulation

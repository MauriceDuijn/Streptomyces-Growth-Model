import numpy as np
from utils.BenchmarkTimer import Timer
from utils.Animator import CellGrowthAnimator
from Chemistry_manager.ReactionChannel import Reaction
from Event_manager.Condition import Condition
from Event_manager.Event import Event
from Cell_manager.Cell import Cell


class GillespieSimulator:
    benchmark = Timer()

    def __init__(self,
                 end_time: float,
                 reaction_channels: list[Reaction],
                 conditions: list[Condition],
                 events: list[Event],
                 cells: list[Cell],
                 animator: CellGrowthAnimator = None):
        self.end_time: float = end_time
        self.reactions: list[Reaction] = reaction_channels
        self.conditions: list[Condition] = conditions
        self.events: list[Event] = events
        self.cells: list[Cell] = cells

        # Initialize parameters start of simulation
        self.total_propensity = 0
        self.total_events = 0
        self.run_time = 0
        self.tau = 0
        self.next_log = 0
        self.log_interval = 1   # Changes the amount of logs per time unit

        self.animator = animator

    def run(self):
        print("[Start Gillespie algorithm]")
        while self.run_time < self.end_time:
            self._update_propensities()

            if self.total_propensity == 0:
                print("[No reactions left]")
                break

            self._update_time_increment()
            self._increment_continuous_factors()

            if self.run_time >= self.end_time:
                print("[End time reached]")
                break

            self._execute_event()

            # if self.total_events == 300:
            #     print("[Max events reached]")
            #     break

            self._log_data()

        self._end_of_simulation()

    def _update_propensities(self):
        self._update_reaction_base_propensity()
        self._update_condition_factors()
        self._update_event_propensities()
        self._apply_state_mask()
        self._update_total_propensity()

    @benchmark.measure_decorator("reaction_base_propensity")
    def _update_reaction_base_propensity(self):
        for reaction in self.reactions:
            reaction.calc_propensity()

    @benchmark.measure_decorator("condition_factors")
    def _update_condition_factors(self):
        for condition in self.conditions:
            condition.calc_factor()

    @benchmark.measure_decorator("event_propensities")
    def _update_event_propensities(self):
        for event in self.events:
            event.update_propensity()

    @staticmethod
    @benchmark.measure_decorator("state_mask")
    def _apply_state_mask():
        Event.state_mask_correction()

    @benchmark.measure_decorator("total_propensity")
    def _update_total_propensity(self):
        Event.update_total_propensity()
        self.total_propensity = Event.total_propensity

    def _no_reactions_left(self):
        return self.total_propensity == 0

    @benchmark.measure_decorator("time increase")
    def _update_time_increment(self):
        self.tau = np.random.exponential(1 / self.total_propensity)
        self.run_time += self.tau

    @benchmark.measure_decorator("time check")
    def _next_reaction_passes_end_time(self):
        return self.run_time >= self.end_time

    @benchmark.measure_decorator("continuous factors")
    def _increment_continuous_factors(self):
        Cell.increase_age_over_time(self.tau)
        Cell.increase_polarisome_over_time(self.tau)

    @benchmark.measure_decorator("event execution")
    def _execute_event(self):
        cell_index, event_index = self._pick_random_cell_and_event()
        self.events[event_index].update(self.cells[cell_index])
        self.total_events += 1

    @staticmethod
    def _pick_random_cell_and_event():
        return Event.random_cell_event_index()

    @benchmark.measure_decorator("log data")
    def _log_data(self):
        while self.run_time > self.next_log:
            print(self.run_time, self.total_propensity)
            self.next_log += self.log_interval

        if self.animator:
            self.animator.snapshot_schedule(self.run_time)

    def _end_of_simulation(self):
        print("[Total events]", self.total_events)
        self.benchmark.print_times()

        # if self.animator:
        #     self.animator.render(save_path="Tropism_test.1.mp4", overwrite=True)


# def gillespie_algorithm(end_time: float,
#                         reaction_channels: list[Reaction],
#                         conditions: list[Condition],
#                         events: list[Event],
#                         cells: list[Cell]):
#     btimer = Timer()
#     event_counter = 0
#     run_time = 0
#     print("[Start Gillespie algorithm]")
#     while run_time < end_time:
#         # Update propensity values
#         for reaction in reaction_channels:
#             reaction.calc_propensity()
#
#         for condition in conditions:
#             condition.calc_factor()
#
#         for event in events:
#             event.update_propensity()
#
#         # Apply the mask
#         Event.state_mask_correction()
#
#         # Calculate the total propensity
#         Event.update_total_propensity()
#         A0 = Event.total_propensity
#
#         # Stop if there are no reactions left
#         if A0 == 0:
#             print("[No reactions left]")
#             break
#
#         # Update the time
#         tau = np.random.exponential(1 / A0)
#         run_time += tau
#
#         # Increase continuous parameters
#         Cell.increase_age_over_time(tau)
#         Cell.increase_polarisome_over_time(tau)
#
#         # Stop if the next event happens
#         if run_time >= end_time:
#             print("[End time reached]")
#             break
#
#         # Pick a cell and event index based on the propensity
#         cell_index, event_index = Event.random_cell_event_index()
#         events[event_index].update(cells[cell_index])
#         event_counter += 1
#
#         # print(run_time, cell_index, events[event_index].name)
#         # print(run_time, A0, Reaction.reaction_channels[0].propensity, Cell.total_cells, *[elem.amount for elem in Element.elemental_species], event_counter)
#
#     # Testing
#     print(Cell.DivIVA_array[Cell.DivIVA_array.active > 0])
#
#     # Outside loop stats
#     print(event_counter)
#     btimer.print_times()

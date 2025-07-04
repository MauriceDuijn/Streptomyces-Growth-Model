import numpy as np
from src.utils.benchmark_timer import Timer
from src.utils.analysis.simulation_logger import SimulationLogger
from src.utils.analysis.report_manager import ReportManager
from src.utils.visual.colony_structure_plotter import ColonyPlotter
from src.utils.visual.animator import CellGrowthAnimator
from src.algorithm.chemistry.reaction import Reaction
from src.algorithm.event.condition import Condition
from src.algorithm.event.event import Event
from src.algorithm.cell_based.cell import Cell

from src.algorithm.cell_based.cell_action import action_timer


class GillespieSimulator:
    """
    Implementation of the Gillespie algorithm for handeling stochastic discrete cell based events.
    Events are influenced by reaction and conditions.
    Simulation ends when the end time is reached or when there are no events left.
    """
    benchmark = Timer()

    def __init__(self,
                 end_time: float,
                 reaction_channels: list[Reaction],
                 conditions: list[Condition], events: list[Event],
                 cells: list[Cell],
                 logger: SimulationLogger = None, reporter: ReportManager = None,
                 plotter: ColonyPlotter = None, animator: CellGrowthAnimator = None):
        self.end_time: float = end_time
        self.reactions: list[Reaction] = reaction_channels
        self.conditions: list[Condition] = conditions
        self.events: list[Event] = events
        self.cells: list[Cell] = cells
        self.logger: SimulationLogger = logger.set_simulator(self) if logger else logger
        self.reporter: ReportManager = reporter
        self.plotter: ColonyPlotter = plotter
        self.animator: CellGrowthAnimator or None = animator

        # Initialize parameters start of simulation
        self.total_propensity: float = 0
        self.total_events: int = 0
        self.run_time: float = 0
        self.tau: float = 0

    def run(self):
        """Main loop of the Gillespie algorithm."""
        print("[Start Gillespie algorithm]")
        while self.run_time < self.end_time:
            self._update_propensities()

            if self.total_propensity == 0:
                self._end_of_simulation("No reactions left")
                break

            self._update_time_increment()
            self._increment_continuous_factors()

            if self.run_time >= self.end_time:
                self._end_of_simulation("End time reached")
                break

            self._execute_event()
            self._log_data()

    def _update_propensities(self):
        """Change and update all propensity related effects."""
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
        self.total_propensity = Event.get_total_propensity()

    @benchmark.measure_decorator("time increase")
    def _update_time_increment(self):
        """Increase the time based on the systems total propensity."""
        self.tau = np.random.exponential(1 / self.total_propensity)
        self.run_time += self.tau

    @benchmark.measure_decorator("time check")
    def _next_reaction_passes_end_time(self):
        return self.run_time >= self.end_time

    @benchmark.measure_decorator("continuous factors")
    def _increment_continuous_factors(self):
        """Increase time related events."""
        Cell.increase_age_over_time(self.tau)
        Cell.increase_polarisome_over_time(self.tau)

    @benchmark.measure_decorator("event execution")
    def _execute_event(self):
        """Pick a random weighted event and execute its action."""
        cell_index, event_index = Event.random_cell_event_index(self.total_propensity)
        self.events[event_index].update(self.cells[cell_index])
        self.total_events += 1

    @benchmark.measure_decorator("log repeat_data")
    def _log_data(self):
        """Store repeat_data in logger, reporter and animator based on set configs."""
        if self.logger:
            self.logger.log(self.run_time)

        if self.reporter:
            self.reporter.report(self.run_time)

        if self.animator:
            self.animator.snapshot_schedule(self.run_time)

    def _end_of_simulation(self, end_condition: str):
        """Final actions."""
        self._log_data()
        print(f"[End simulation: {end_condition}]")
        print("[Total events]", self.total_events)
        self.benchmark.print_times()

        action_timer.print_times()

        # if self.animator:
        #     self.animator.render(save_path="Tropism_test.1.mp4", overwrite=True)

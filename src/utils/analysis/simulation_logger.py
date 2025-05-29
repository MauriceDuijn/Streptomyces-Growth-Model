from src.configs.load_config import LoggerConfig
from src.algorithm.chemistry.element import Element
from src.algorithm.chemistry.reaction import Reaction
from src.algorithm.event.state import State


class LogState:
    """Container for all logged simulation repeat_data at one time point"""
    run_time: float
    element_count: list[int]
    reaction_propensities: list[float]
    state_counts: list[int]
    log_total_propensities: bool = True
    log_events_total_propensity: bool = True

    def set_time(self, run_time: float):
        self.run_time = run_time

    def set_element_count(self):
        self.element_count = [element.amount for element in Element.instances]

    def set_reaction_propensities(self):
        self.reaction_propensities = [reaction.propensity.astype(float) for reaction in Reaction.instances]

    def set_state_counts(self):
        self.state_counts = [state.count for state in State.instances]


class SimulationLogger:
    def __init__(self, config, log_interval: float = 1.0):
        self.config: LoggerConfig = config
        self.log_states: list[LogState] = []
        self.next_log: float = 0.0
        self.log_interval: float = log_interval
        self.simulator = None

    def log(self, run_time):
        """Capture the current simulation state"""
        while self.should_log(run_time):
            # Simple print statement
            print(f"T: {run_time: <20}A0: {self.simulator.total_propensity}")
            self.increment_log_timing()
            self.make_log(run_time)

    def make_log(self, run_time):
        log_state = self.capture_state(run_time)
        self.log_states.append(log_state)

    def capture_state(self, run_time):
        log_state = LogState()

        log_state.set_time(run_time)
        if self.config.log_element_count:
            log_state.set_element_count()

        if self.config.log_reaction_propensity:
            log_state.set_reaction_propensities()

        return log_state

    def should_log(self, run_time) -> bool:
        """Check if it's time to log based on run_time and log_interval"""
        return run_time >= self.next_log

    def increment_log_timing(self):
        """Increase the next logged time point"""
        self.next_log += self.log_interval

    def set_simulator(self, simulator):
        """Set the simulator reference."""
        self.simulator = simulator
        return self


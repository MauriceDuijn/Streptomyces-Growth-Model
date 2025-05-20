import json
import pandas as pd
import numpy as np
from pathlib import Path

from Chemistry_manager.ElementalSpecies import Element
from Chemistry_manager.ReactionChannel import Reaction
from Event_manager.State import State


class LoggerConfig:
    log_element_counts: bool = True
    log_reaction_propensities: bool = True
    log_state_counts: bool = True
    log_total_propensities: bool = True
    log_events_total_propensity: bool = True


class LogState:
    """Container for all logged simulation data at one time point"""
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
        self.state_counts = [state.count for state in State.state_nodes]


class SimulationLogger:
    def __init__(self, config, log_interval: float = 1.0):
        self.config: LoggerConfig = config
        self.log: list[LogState] = []
        self.next_log: float = 0.0
        self.log_interval: float = log_interval
        self.simulator = None

    def log(self):
        """Capture the current simulation state"""
        if not self.should_log():
            return

        self.increment_log_timing()

        log_state = self.capture_state()
        self.log.append(log_state)

    def capture_state(self):
        log_state = LogState()

        log_state.set_time(self.simulator.run_time)

        if self.config.log_element_counts:
            log_state.set_element_count()

        if self.config.log_reaction_propensities:
            log_state.set_reaction_propensities()

        return log_state

    def should_log(self) -> bool:
        """Check if it's time to log based on run_time and log_interval"""
        return self.simulator.run_time >= self.next_log

    def increment_log_timing(self):
        """Increase the next logged time point"""
        self.next_log += self.log_interval

    def set_simulator(self, simulator):
        self.simulator = simulator

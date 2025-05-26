import numpy as np
from pathlib import Path
from src.utils.Analysis.ColonyReport import ColonyAnalysisReport
from src.organic.colony import Colony


class ReportManager:
    def __init__(self, time_points: list[float], report_params: list[str] = None, save_as_json=False):
        self.reports: list[ColonyAnalysisReport] = []
        self.time_points: list[float] = time_points
        self.tp_tracker: int = 0
        self.next_tp: float = time_points[self.tp_tracker]
        self.save_as_json: bool = save_as_json

        self.simulator = None

        if report_params:
            ColonyAnalysisReport.report_params = report_params

    def report(self):
        while self.should_report():
            self.increment_timing()
            self.make_report()

    def make_report(self):
        report = ColonyAnalysisReport(Colony.colonies)
        report.run_analysis()
        self.reports.append(report)

    def should_report(self):
        return self.simulator.run_time >= self.next_tp

    def increment_timing(self):
        self.tp_tracker += 1

        if self.tp_tracker == len(self.time_points):
            # Prevents out of range error for the last time point
            self.next_tp = np.inf
        else:
            self.next_tp = self.time_points[self.tp_tracker]

    def set_simulator(self, simulator):
        self.simulator = simulator
        return self

    def write_reports_to_run_file(self, run_dir: Path, repeat_index: int):
        repeat_file_path: Path = run_dir / f"Repeat_{repeat_index}"
        for report in self.reports:
            time_point: float = self.time_points[report.index]
            if self.save_as_json:
                report.save_as_json(repeat_file_path, time_point)
            else:
                report.save_as_formatted_report(repeat_file_path, time_point)

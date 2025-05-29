import json
import numpy as np
from pathlib import Path
from src.configs.load_config import ReporterConfig
from src.utils.visual.report_plotter import ReportPlotter
from src.utils.analysis.colony_report import ColonyAnalysisReport
from src.algorithm.cell_based.colony import Colony


class ReportManager:
    def __init__(self, report_config: ReporterConfig):
        self.reports: list[ColonyAnalysisReport] = []
        self.time_points: list[float] = report_config.TIME_POINTS
        self.next_tp: float = self.time_points[0]
        self.tp_tracker: int = 0
        self.save_as_json: bool = report_config.SAVE_AS_JSON_FORMAT
        ColonyAnalysisReport.report_params = report_config.ACTIVE_PARAMETERS

    def report(self, run_time):
        """Generate reports at configured time points."""
        while self.should_report(run_time):
            self.increment_timing()
            self.make_report()

    def make_report(self):
        """Create a new analysis report."""
        report = ColonyAnalysisReport(Colony.instances)
        report.run_analysis()
        self.reports.append(report)

    def should_report(self, run_time):
        return run_time >= self.next_tp

    def increment_timing(self):
        """Update the next reporting time point."""
        self.tp_tracker += 1

        if self.tp_tracker == len(self.time_points):
            # Prevents out of range error for the last time point
            self.next_tp = np.inf
        else:
            self.next_tp = self.time_points[self.tp_tracker]

    def write_reports_to_run_file(self, repeat_file_path: Path):
        """Save all reports to the repeat file."""
        for report in self.reports:
            time_point: float = self.time_points[report.index]
            if self.save_as_json:
                report.save_as_json(repeat_file_path, time_point)
            else:
                report.save_as_formatted_report(repeat_file_path, time_point)

    @staticmethod
    def load_config(run_directory: Path) -> dict:
        """Load configuration from a run directory."""
        config_path = run_directory / "configs.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found at: {config_path}\n"
                f"Run directory contents: {list(run_directory.glob('*'))}"
            )

        with open(config_path, 'r') as config_file:
            return json.load(config_file)

    @staticmethod
    def load_run_data(run_directory: Path) -> dict[str, list[dict]]:
        """Load all repeat data from a run directory."""
        run_data: dict[str, list[dict]] = {}
        repeat_dirs = run_directory.glob("Repeat_*")
        # repeat_file_names: map = map(str, run_directory.glob("Repeat_*.jsonl"))

        for repeat_dir in repeat_dirs:
            report_file = repeat_dir / repeat_dir.name
            with open(report_file.with_suffix(".jsonl"), 'r') as run_file:
                reports: list[dict] = [json.loads(report) for report in run_file]
                run_data[repeat_dir.name] = reports
        return run_data

    @classmethod
    def create_plotter(cls, run_directory: Path) -> ReportPlotter:
        """
        Factory method to create a ReportPlotter with loaded data.
        Can only make a plotter at the end of a run deu to unsaved repeat data.
        """
        config = cls.load_config(run_directory)
        run_data = cls.load_run_data(run_directory)
        return ReportPlotter(config, run_data)

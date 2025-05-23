import json
import os
from pathlib import Path
from utils.RunManager import RunManager


class ReportPlotter:
    def __init__(self, run_manager: RunManager):
        self.rm = run_manager
        self.run_data: dict[str, list[dict]] = {}

    def load_json(self):
        run_file_names: list[str] = os.listdir(self.rm.current_run_dir)
        for file_name in run_file_names:
            run_path: Path = self.rm.current_run_dir / file_name
            if file_name == "config_settings":
                continue
            with open(run_path, 'r') as run_file:
                for report in run_file:
                    report_data: dict = json.loads(report)
                    reports = self.run_data.get(file_name, [])
                    reports.append(report_data)
                    self.run_data[file_name] = reports

    def show_run_data(self):
        for run, data in self.run_data.items():
            print(run, ":")
            for report in data:
                print(report["Report_ID"], report["Time_Point"])
                print(report)

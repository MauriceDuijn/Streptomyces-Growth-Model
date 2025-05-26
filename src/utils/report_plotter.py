import json
import os
from pathlib import Path
import matplotlib.pyplot as plt


class ReportPlotter:
    def __init__(self, run_directory: Path):
        self.run_dir: Path = run_directory
        self.config: dict[str, dict] = {}
        self.run_data: dict[str, float or dict] = {}

    def plot_run(self):
        if not self.run_data:
            # Load in run repeat_data
            self.load_json()

        for repeat, data in self.run_data.items():
            self.plot_repeat(data)

    def load_json(self):
        """Load in json formatted repeat_data."""
        self.load_config()
        self.load_run_data()

    def load_config(self):
        """Load in config repeat_data."""
        config_path: Path = self.run_dir / "config.json"
        with open(config_path, 'r') as config_file:
            full_config = json.load(config_file)
            self.config = full_config["configs"]

    def load_run_data(self):
        """Load in repeat repeat_data."""
        repeat_file_names: list[str] = os.listdir(self.run_dir)
        for file_name in repeat_file_names:
            if file_name in ["config", "config.json"]:
                # Skip the config files
                continue
            repeat_path: Path = self.run_dir / file_name
            with open(repeat_path, 'r') as run_file:
                for report in run_file:
                    report_data: dict = json.loads(report)
                    reports = self.run_data.get(file_name, [])
                    reports.append(report_data)
                    self.run_data[file_name] = reports

    def plot_repeat(self, repeat_data):
        time_points = self.config["ReporterConfig"]["TIME_POINTS"]
        parameter_names: list[str] = self.config["ReporterConfig"]["ACTIVE_PARAMETERS"]

        for param in parameter_names:
            plt.figure(figsize=(10, 6))

            violin_data = [
                report["Parameters"][param]
                for report in repeat_data
            ]

            base_width = 0.9
            time_interval = self.config["GlobalConfig"]["REPORT_INTERVAL"]
            base_width *= time_interval

            plt.violinplot(
                violin_data,
                widths=base_width,
                positions=time_points,
                showmeans=False,
                showmedians=False,
                showextrema=False
            )

            plt.title(f"Parameter: {param}")
            plt.xlabel("Time Point")
            plt.ylabel("Value")
            # plt.legend(title="Sample")
            plt.show()


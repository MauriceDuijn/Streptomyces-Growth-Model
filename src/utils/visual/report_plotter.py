import json
from pathlib import Path
import matplotlib.pyplot as plt


class ReportPlotter:
    def __init__(self, config: dict, run_data: dict[str, list[dict]]):
        self.config: dict[str, dict] = config["configs"]
        self.run_data: dict[str, list[dict]] = run_data

    def plot_run(self, repeat_index: int = None):
        """Plot either all repeats or a specific repeat if index is provided."""
        for repeat, data in self.run_data.items():
            if not repeat_index or str(repeat_index) == repeat.removesuffix(".jsonl")[-1]:
                self.plot_repeat(data)

    def plot_repeat(self, repeat_data: list[dict]):
        """Plot all parameters and metrics for a single repeat."""
        parameter_names: list[str] = self.config["ReporterConfig"]["ACTIVE_PARAMETERS"]
        metric_names: list[str] = self.config["ReporterConfig"]["ACTIVE_METRICS"]
        time_interval = self.config["ReporterConfig"]["REPORT_INTERVAL"]

        base_width = 0.9 * time_interval
        time_points = [report["Time_Point"] for report in repeat_data]

        # Plot all active parameters
        for param in parameter_names:
            plt.figure(figsize=(10, 6))
            violin_data = [report["Parameters"][param] for report in repeat_data]
            self._plot_violin(violin_data, base_width, time_points)
            plt.title(f"Parameter: {param}")
            plt.xlabel("Time Point")
            plt.ylabel("Value")

        # Plot all the set metrics
        for metric in metric_names:
            plt.figure(figsize=(10, 6))
            violin_data = [report["Parameters"][metric] for report in repeat_data]
            self._plot_violin(violin_data, base_width, time_points)
            plt.title(f"Metric: {metric}")
            plt.xlabel("Time Point")
            plt.ylabel("Value")

        plt.show()

    @staticmethod
    def _plot_violin(violin_data: list, width: float, positions: list[float]):
        violin_parts = plt.violinplot(
            violin_data,
            widths=width,
            positions=positions,
            showmeans=True,
            showmedians=False,
            showextrema=False
        )

        # Add borders to each violin
        for pc in violin_parts['bodies']:
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)

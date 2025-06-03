import matplotlib.pyplot as plt


class ReportPlotter:
    param_units = {
        "min_diameter": "µm",
        "max_diameter": "µm",
        "number_of_cells": "amount",
        "num_active_cells": "amount",
        "max_crowding": "crowding index",
        "average_crowding": "crowding index",
        "average_propensity": "propensity",
        "area": "µm^2",
        "log10_area": "µm^2",
        "total_length": "µm",
        "hyphal_density": "µm/µm^2",
        "tip_density": "#tips/µm"
    }

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

        def plot_category(category_type: str, names: list[str]):
            for name in names:
                # print(name, self.param_units[name])
                plt.figure(figsize=(10, 6))
                violin_data = [report["Parameters"][name] for report in repeat_data]
                self._plot_violin(violin_data[1:], base_width, time_points[1:])
                formatted_name = name.replace("_", " ").title()
                plt.title(f"{category_type.title()}: {formatted_name}", size=24)
                plt.xlabel("Time Point", size=18)
                plt.ylabel(self.param_units[name], size=18)
                plt.tick_params(axis='both', which='major', labelsize=14)
                plt.ylim(bottom=0)

        plot_category("Parameter", parameter_names)
        plot_category("Metric", metric_names)
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

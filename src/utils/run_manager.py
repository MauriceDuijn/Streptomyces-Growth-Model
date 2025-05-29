from datetime import datetime, timedelta
from time import perf_counter
import sys
import os
import json
import warnings
from pathlib import Path
from importlib.metadata import version
from src.utils.cell_data_manager import CellDataManager
from src.configs.load_config import Config
from src.algorithm.simulation_init import init
from src.utils.analysis.report_manager import ReportManager
from src.utils.analysis.colony_report import ColonyAnalysisReport
from src.utils.visual.report_plotter import ReportPlotter


class RunManager:
    # List of configs that get stored
    CONFIGS: list = [
        Config().run,
        Config().cell,
        Config().diviva,
        Config().chem,
        Config().report
    ]

    def __init__(self, project_root: Path):
        config = Config()
        self.name: str = config.run.RUN_NAME
        self.project_root: Path = Path(project_root)
        self.current_run_dir: Path = self.project_root / "data" / "runs" / config.run.RUN_NAME
        self.repeats: int = config.run.RUN_REPEATS
        self.plotter: ReportPlotter or None = None

        self.run_start: float = 0
        self.run_total_time: float = 0

    def start(self):
        """
        Start simulating multiple repeated runs based on the initializer.
        Automatically creates a run directory with analysis reports and run configs.
        """
        self.make_run_directory()

        # Start run
        self.run_start = perf_counter()
        for repeat_index in range(1, self.repeats + 1):
            # Run simulation
            simulation = init()
            simulation.run()
            self.save_data(simulation, repeat_index)
        self.run_total_time = perf_counter() - self.run_start

        # Save config
        if self.name != "":
            self.save_config()
            self.save_config_json()

    def save_data(self, simulation, repeat_index):
        if not self.name == "":
            repeat_path = self.get_repeat_path(repeat_index)
            self.save_report(simulation.reporter, repeat_path)
            self.save_simulation_state(repeat_path)

    def get_repeat_path(self, repeat_index):
        repeat_dir = self.current_run_dir / f"Repeat_{repeat_index}"
        os.makedirs(repeat_dir)
        return self.current_run_dir / f"Repeat_{repeat_index}" / f"Repeat_{repeat_index}"

    def plot_run(self, repeat_index=None):
        if self.name == "":
            return warnings.warn("No run data available. Run name is empty.\n"
                                 "Replace the run name for existing run and then plot.")

        self.plotter = ReportManager.create_plotter(self.current_run_dir)
        self.plotter.plot_run(repeat_index=repeat_index)

    def make_run_directory(self):
        """
        Create a new directory for data storage of the current run.
        In this directory stores:
        - Config parameters
        - Run repeat reports
        - Run repeat cell and colony data
        """
        if self.name == "":
            # Skip when no run name
            return

        os.makedirs(Path(self.project_root) / "data" / "runs", exist_ok=True)
        try:
            os.makedirs(self.current_run_dir)
        except FileExistsError as error:
            raise FileExistsError(
                f"{error}\n"
                f"Error: Run directory '{self.name}' already exists.\n\n"
                f"[Tip] Make sure that the run name is unique or delete the existing folder.\n"
                f"Look in '{os.path.abspath(f'{self.project_root}/runs')}' for existing run names.\n"
            )

    def save_config(self):
        """
        Save the used configs parameters and values used in the current run.
        The file is written in a more simple readable format.
        Also stores meta-data of the versions of the requirements.

        Update the RunManager.CONFIGS if any new configs classes need to be saved in the file.
        """
        config_file: Path = self.current_run_dir / "configs"
        with open(config_file, 'w') as file:
            file.write(f"[{self.name}]\n")
            time_stamp: str = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
            file.write(f"Run start date and time: {time_stamp}\n")
            file.write(f"Total run time: {timedelta(seconds=self.run_total_time)}\n")

            for config in self.CONFIGS:
                file.write(f"\n")
                file.write(f"{config.__class__.__name__}:\n")
                class_variables = self.get_class_vars(config)
                for name, value in class_variables.items():
                    file.write(f"\t{name}: {value}\n")

            file.write("\nVersions\n")
            file.write(f"\tPython: {sys.version.split()[0]}\n")
            file.write(f"\tnumpy: {version('numpy')}\n")
            file.write(f"\tmatplotlib: {version('matplotlib')}\n")
            file.write(f"\tscipy: {version('scipy')}\n")

    def save_config_json(self):
        """
        Save the used configs parameters and values used in the current run in json format.
        Used by ReportPlotter class.
        Functions the same as the default save_config()
        """
        config_file: Path = self.current_run_dir / "configs.json"
        config_data: dict = {
            "meta_data": {
                "Run_name": self.name,
                "Run_start": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "Total_run_time": self.run_total_time,
            },
            "configs": {
                config.__class__.__name__: {
                    name:  value
                    for name, value in self.get_class_vars(config).items()
                }
                for config in self.CONFIGS
            }
        }
        with open(config_file, 'w') as jfile:
            json.dump(config_data, jfile)

    @staticmethod
    def save_report(reporter: ReportManager, repeat_path: Path):
        """
        Saves report to a run repeat file.
        Also clears the stored reports for next repeat.

        :param reporter: ReportManager object of the current repeated run
        :param repeat_path: Data storage path for the reports
        """
        reporter.write_reports_to_run_file(repeat_path)
        ColonyAnalysisReport.reset_class()

    @staticmethod
    def save_simulation_state(repeat_path: Path):
        """
        Stores the end state of the cells and colonies.
        Stored data can be loaded in for extending a run.

        :param repeat_path: Data storage path for the reports
        """
        cdm = CellDataManager(repeat_path)
        cdm.save_cell_simulation_data()

    @staticmethod
    def get_class_vars(config_class):
        """
        Automatically format all set parameters and values in a configs class.
        Filters all non-parameter related class variables returned from vars().

        :param config_class: Config class with parameters and values
        :return: Dict of parameter of a configs class
        """
        return {name: value
                for name, value in vars(config_class).items()
                if not name.startswith("__")
                and not callable(value)
                and not isinstance(value, (classmethod, staticmethod, property))
                }

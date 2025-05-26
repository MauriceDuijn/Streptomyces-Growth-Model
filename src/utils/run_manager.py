import sys
from time import perf_counter
from datetime import datetime
import json
import os
from pathlib import Path
from src.init.configs import GlobalConfig, CellConfig, DivIVAConfig, ChemicalConfig, ReporterConfig
from src.init.simulation_init import init
from src.utils.Analysis.ReportManager import ReportManager
from src.utils.Analysis.ColonyReport import ColonyAnalysisReport
from src.utils.report_plotter import ReportPlotter


class RunManager:
    PROJECT_ROOT: Path
    CONFIGS: list = [
        GlobalConfig,
        CellConfig,
        DivIVAConfig,
        ChemicalConfig,
        ReporterConfig
    ]

    def __init__(self):
        self.name: str = GlobalConfig.RUN_NAME
        self.current_run_dir: Path = Path(self.PROJECT_ROOT) / "runs" / GlobalConfig.RUN_NAME
        self.repeats: int = GlobalConfig.RUN_REPEATS
        self.plotter: ReportPlotter = ReportPlotter(self.current_run_dir)

        self.run_start: float = 0
        self.run_total_time: float = 0

    def start(self):
        # self.make_run_directory()

        self.run_start = perf_counter()
        for repeat_index in range(self.repeats):
            plotter, animator, simulation = init()
            simulation.run()
            self.save_report(simulation.reporter, repeat_index)
        self.run_total_time = self.run_start - perf_counter()

        self.save_config()
        self.save_config_json()

    def make_run_directory(self):
        os.makedirs(Path(self.PROJECT_ROOT) / "runs", exist_ok=True)
        try:
            os.makedirs(self.current_run_dir)
        except FileExistsError as error:
            raise FileExistsError(
                f"{error}\n"
                f"Error: Run directory '{self.name}' already exists.\n\n"
                f"[Tip] Make sure that the run name is unique or delete the existing folder.\n"
                f"Look in '{os.path.abspath(f'{self.PROJECT_ROOT}/runs')}' for existing run names.\n"
            )

    def save_report(self, reporter: ReportManager, repeat_index: int):
        reporter.write_reports_to_run_file(self.current_run_dir, repeat_index)
        # Clean the stored analysis reports
        ColonyAnalysisReport.reset_class()

    def save_config(self):
        config_file: Path = self.current_run_dir / "config"
        with open(config_file, 'w') as file:
            file.write(f"[{self.name}]\n")
            time_stamp: str = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
            file.write(f"Run start date and time: {time_stamp}\n")
            file.write(f"Total run time: {self.run_total_time}")

            for config in self.CONFIGS:
                file.write(f"\n")
                file.write(f"{config.__name__}:\n")
                class_variables = self.get_class_vars(config)
                for name, value in class_variables.items():
                    file.write(f"\t{name}: {value}\n")

    def save_config_json(self):
        config_file: Path = self.current_run_dir / "config.json"
        config_data: dict = {
            "meta_data": {
                "Run_name": self.name,
                "Run_start": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "Total_run_time": self.run_total_time,
            },
            "configs": {
                config.__name__: {
                    name:  value
                    for name, value in self.get_class_vars(config).items()
                }
                for config in self.CONFIGS
            },
            "Environment": {
                "Python_version": sys.version.split()[0],
                "OS_info": {
                    "Name": os.name,
                    "Seperator": os.sep,
                    "Current_work_directory": os.curdir
                },
                "Dependecies": {
                    ""
                }
            }
        }
        with open(config_file, 'w') as jfile:
            json.dump(config_data, jfile)

    @staticmethod
    def get_class_vars(target_class):
        return {name: value
                for name, value in vars(target_class).items()
                if not name.startswith("__")
                and not callable(value)
                and not isinstance(value, (classmethod, staticmethod, property))
                }

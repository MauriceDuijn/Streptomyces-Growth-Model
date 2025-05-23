from datetime import datetime
import os
from pathlib import Path
from init.configs import GlobalConfig, CellConfig, DivIVAConfig, ChemicalConfig
from init.new_init import init
from Analysis.ReportManager import ReportManager
from Analysis.ColonyReport import ColonyAnalysisReport


class RunManager:
    PROJECT_ROOT: Path
    CONFIGS: list = [
        GlobalConfig,
        CellConfig,
        DivIVAConfig,
        ChemicalConfig
    ]

    def __init__(self):
        self.name: str = GlobalConfig.RUN_NAME
        self.current_run_dir: Path = Path(self.PROJECT_ROOT) / "runs" / GlobalConfig.RUN_NAME
        self.repeats: int = GlobalConfig.RUN_REPEATS

    def start(self):
        self.make_run_directory()
        self.store_config()

        for run_number in range(self.repeats):
            plotter, animator, simulation = init()
            simulation.run()
            self.save_report(simulation.reporter, run_number)

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

    def store_config(self):
        config_file: str = os.path.join(self.current_run_dir, "config_settings")
        with open(config_file, 'w') as file:
            file.write(f"[{self.name}]\n")
            time_stamp: str = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
            file.write(f"Run start date and time: {time_stamp}\n")

            for config in self.CONFIGS:
                file.write(f"\n")
                file.write(f"{config.__name__}:\n")
                class_variables = self.get_class_vars(config)
                for name, value in class_variables.items():
                    file.write(f"\t{name}: {value}\n")

    def save_report(self, reporter, run_number):
        reporter.write_reports_to_run_file(self.current_run_dir, run_number)
        # Clean the stored analysis reports
        ColonyAnalysisReport.reset_class()

    @staticmethod
    def get_class_vars(target_class):
        return {name: value
                for name, value in vars(target_class).items()
                if not name.startswith("__")
                and not callable(value)
                and not isinstance(value, (classmethod, staticmethod, property))
                }

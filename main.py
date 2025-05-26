from time import perf_counter
from pathlib import Path
from src.utils import RunManager


def main() -> None:
    PROJECT_ROOT = Path(__file__).parent
    RunManager.PROJECT_ROOT = PROJECT_ROOT
    RM = RunManager()
    RM.start()                  # Run a simulation run based on configs
    RM.plotter.plot_run()       # Show generated repeat_data


if __name__ == '__main__':
    start = perf_counter()
    main()
    print("Total code time", perf_counter() - start)

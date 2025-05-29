from time import perf_counter
from pathlib import Path
from src.utils.run_manager import RunManager


def main() -> None:
    # Define project root relative to THIS file (main.py)
    project_root_directory = Path(__file__).parent

    # Initialize RunManager
    run_manager = RunManager(project_root_directory)
    # Run a simulation run based on configs
    run_manager.start()

    # Show run analysis plots
    while (answer := input("Do you want to plot the current run? [Y/N]")).lower() not in ["y", "n"]:
        print("Invalid input. Type in single character.")

    if answer.lower() == "y":
        run_manager.plot_run(1)

    # run_manager.initialize_simulation()
    # run_manager.load_repeat_data(run_manager.get_repeat_path(0))
    # from src.algorithm.cell_based.colony import Colony
    # ColonyPlotter().plot_cells_2D_colony(Colony.instances[0])


if __name__ == '__main__':
    start = perf_counter()
    main()
    print("Total code time", perf_counter() - start)

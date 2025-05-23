from time import perf_counter
from pathlib import Path
from utils.RunManager import RunManager
from utils.ReportPlotter import ReportPlotter


def main() -> None:
    PROJECT_ROOT = Path(__file__).parent
    RunManager.PROJECT_ROOT = PROJECT_ROOT
    RM = RunManager()
    RM.start()
    RP = ReportPlotter(RM)
    RP.load_json()
    RP.show_run_data()

    # # for i in np.arange(0.1, 1.1, 0.2):
    # for repeat in range(10):
    #     plotter, animator, simulation = init()
    #     simulation.run()
    #     # plotter.plot_cells_2D_colony(Colony.colonies[0], "crowding_index", cmap="nipy_spectral_r", with_hull=True)
    #     # plotter.plot_cells_2D_colony(Colony.colonies[0], "age", cmap="Spectral_r")
    #
    #     # action_timer.print_times()
    #
    #     analysis = ColonyAnalysisReport(Colony.colonies)
    #     analysis.run_analysis()
    #     # analysis.show_plots()
    #
    # ColonyAnalysisReport.meta_analysis()

    # plotter.plot_cells_2D("crowding_index")
    # plotter.plot_cells_2D_colony(Colony.colonies[0], "DivIVA", cmap="viridis", sort_values=True)

    # animator.render(save_path="Tropism_test.interesant.5.mp4", overwrite=True)
    # plotter.plot_cells_2D()
    # plotter.plot_cells_2D("age")
    # plotter.plot_cells_2D("DivIVA", cmap="viridis", sort_values=True)

    # plotter.plot_cells_3D_crowding_index()


if __name__ == '__main__':
    start = perf_counter()
    main()
    print("Total code time", perf_counter() - start)

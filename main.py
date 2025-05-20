from time import perf_counter
import numpy as np
from init.new_init import init, CellConfig
from Cell_manager.Colony import Colony,Cell
from Event_manager.Condition import Condition
from Cell_manager.CellAction import action_timer
from Analysis.Colony_analysis import ColonyAnalysis


def main() -> None:
    # for i in np.arange(0.1, 1.1, 0.2):
    for repeat in range(5):
        plotter, animator, simulation = init()
        simulation.run()
        plotter.plot_cells_2D_colony(Colony.colonies[0], "crowding_index", cmap="nipy_spectral_r")

        action_timer.print_times()

        analysis = ColonyAnalysis(Colony.colonies)
        analysis.run_analysis()
        # analysis.show_plots()

    ColonyAnalysis.meta_analysis()

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

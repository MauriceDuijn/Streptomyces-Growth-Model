from time import perf_counter
from init.new_init import init
from Cell_manager.Colony import Colony,Cell
from Event_manager.Condition import Condition


def main() -> None:
    plotter, animator, simulation = init()
    simulation.run()

    # plotter.plot_cells_2D("crowding_index")
    plotter.plot_cells_2D_colony(Colony.colonies[0], "crowding_index")
    plotter.plot_cells_2D_colony(Colony.colonies[0], "DivIVA", cmap="viridis", sort_values=True)

    # animator.render(save_path="Tropism_test.interesant.5.mp4", overwrite=True)
    # plotter.plot_cells_2D()
    # plotter.plot_cells_2D("age")
    # plotter.plot_cells_2D("DivIVA", cmap="viridis", sort_values=True)

    plotter.plot_cells_3D_crowding_index()


if __name__ == '__main__':
    start = perf_counter()
    main()
    print("Total code time", perf_counter() - start)

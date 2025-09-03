import time
from typing import Callable

import numpy as np

import dynamic_path_2 as dynamic_path
import grid
import neural_solver.neural_solver as neural_solver

import torch
from typing import Optional

non_trivial_path = "data/non-trivial-sudokus.csv"
num_executions = 1


def measure_time(
        label: str,
        num_executions: int,
        grids: list[dict[grid.Coord, grid.Cell]],
        solve: Callable[dict[grid.Coord, grid.Cell],
        dynamic_path.SolutionPathNode]
):
    print(label)

    times = np.zeros(num_executions)

    for execution_idx in range(0, num_executions):
        start = time.perf_counter()

        for grid in grids:
            _ = solve(grid)

        end = time.perf_counter()

        times[execution_idx] = end - start

    mean = 1000 * times.mean()
    num_grids = len(grids)
    per_grid = mean / num_grids

    print(
        f"for {num_grids} it took on average {mean:.2f} ms with std {1000 * times.std():.2f} ms, {per_grid:.2f} ms per grid"
    )


quizes = []
for idx, csv_line in enumerate(open(non_trivial_path, 'r').read().splitlines()[1:]):
    _, quiz_str, _ = csv_line.split(",")

    quiz = grid.str_to_grid(quiz_str)

    quizes.append(quiz)

measure_time(
    label="ordered",
    num_executions=num_executions,
    grids=quizes,
    solve=lambda quiz: dynamic_path.solve_valid_grid(
        grid=quiz,
        find_next_coord_and_value=lambda grid: dynamic_path.find_next_coord_then_select_random_value(
            grid=grid,
            find_next_coord=dynamic_path.find_next_coord_for_ordered
        )
    )
)

measure_time(
    label="random",
    num_executions=num_executions,
    grids=quizes,
    solve=lambda quiz: dynamic_path.solve_valid_grid(
        grid=quiz,
        find_next_coord_and_value=lambda grid: dynamic_path.find_next_coord_then_select_random_value(
            grid=grid,
            find_next_coord=dynamic_path.find_next_coord_for_random
        )
    )
)

measure_time(
    label="smallest allowed",
    num_executions=num_executions,
    grids=quizes,
    solve=lambda quiz: dynamic_path.solve_valid_grid(
        grid=quiz,
        find_next_coord_and_value=lambda grid: dynamic_path.find_next_coord_then_select_random_value(
            grid=grid,
            find_next_coord=dynamic_path.find_next_coord_for_smallest_allowed
        )
    )
)

sudoku_net = neural_solver.SudokuNet(
    feature_maps=456,
    row_col_box_out_features=27,
    dense_out_features=729
)

sudoku_net.load_state_dict(torch.load(f"v2/data/cs/model_weights/99.pth"))


def find_next_coord_and_value_for_neural(some_grid: grid.Grid, model) -> Optional[tuple[grid.Coord, int]]:
    quiz: dict[grid.Coord, int] = {}
    allowed_values: dict[grid.Coord, list[int]] = {}

    has_0 = False
    for coord in grid.all_coords_0_to_80:
        value = some_grid.cells[coord].value
        quiz[coord] = value

        if value == 0:
            has_0 = True

        allowed_values[coord] = some_grid.cells[coord].allowed_values

        if has_0:
            return neural_solver.solve(
                quiz=quiz,
                allowed_values=allowed_values,
                model=model,
            )
        else:
            return None


measure_time(
    label="neural",
    num_executions=num_executions,
    grids=quizes,
    solve=lambda quiz: find_next_coord_and_value_for_neural(
        some_grid=quiz,
        model=sudoku_net
    )
)

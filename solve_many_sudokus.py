import time
from typing import Callable

import numpy as np

import dynamic_path
import grid

non_trivial_path = "./data/non-trivial-sudokus.csv"


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

    mean = times.mean()
    num_grids = len(grids)
    per_grid = 1000 * mean / num_grids

    print(f"for {num_grids} it took {mean:.2f} s with std {times.std():.2f} s, {per_grid:.2f} ms per grid")


quizes = []
for idx, csv_line in enumerate(open(non_trivial_path, 'r').read().splitlines()[1:]):
    _, quiz_str, _ = csv_line.split(",")

    quiz = grid.str_to_grid(quiz_str)

    quizes.append(quiz)

measure_time(
    label="ordered",
    num_executions=10,
    grids=quizes,
    solve=lambda quiz: dynamic_path.solve_valid_grid(
        grid=quiz,
        method=dynamic_path.FillPathCreationMethod.ORDERED
    )
)

measure_time(
    label="smallest allowed",
    num_executions=10,
    grids=quizes,
    solve=lambda quiz: dynamic_path.solve_valid_grid(
        grid=quiz,
        method=dynamic_path.FillPathCreationMethod.SMALLEST_ALLOWED
    )
)

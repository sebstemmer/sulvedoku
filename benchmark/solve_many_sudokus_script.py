import time
from typing import Callable

import numpy as np

from grid.grid import Grid, is_equal, str_to_grid
from neural_solve.solve import neural_guess_strategy
from solve.solve import solve_grid, random_guess_strategy, ordered_guess_strategy, smallest_allowed_guess_strategy

non_trivial_path = "./data/327-non-trivial-sudokus.csv"
num_executions = 100


def measure_time(
        label: str,
        num_executions: int,
        grids: list[Grid],
        solutions: list[Grid],
        solve_grid: Callable[[Grid], Grid]
) -> None:
    print(label)

    times = np.zeros((num_executions, len(grids)))

    for execution_idx in range(0, num_executions):
        for grid_idx, my_grid in enumerate(grids):
            solution: Grid = solutions[grid_idx]

            start = time.perf_counter()
            calculated_solution: Grid = solve_grid(my_grid)
            times[execution_idx, grid_idx] = time.perf_counter() - start

            assert is_equal(grid1=calculated_solution, grid2=solution)

    mean_in_ms = 1000 * times.mean()
    std_in_ms = 1000 * times.std()
    num_grids = len(grids)

    print(
        f"for {num_grids} grids it took on average {mean_in_ms:.2f} ms with std {std_in_ms:.2f} ms"
    )


quizes: list[Grid] = []
solutions: list[Grid] = []
for idx, csv_line in enumerate(open(non_trivial_path, 'r').read().splitlines()[1:]):
    _, quiz_str, sol_str = csv_line.split(",")

    quiz: Grid = str_to_grid(quiz_str)
    solution: Grid = str_to_grid(sol_str)

    quizes.append(quiz)
    solutions.append(solution)

measure_time(
    label="ordered",
    num_executions=num_executions,
    grids=quizes,
    solutions=solutions,
    solve_grid=lambda grid: solve_grid(
        grid=grid,
        guess_strategy=ordered_guess_strategy
    )
)

measure_time(
    label="random",
    num_executions=num_executions,
    grids=quizes,
    solutions=solutions,
    solve_grid=lambda grid: solve_grid(
        grid=grid,
        guess_strategy=random_guess_strategy
    )
)

measure_time(
    label="smallest allowed",
    num_executions=num_executions,
    grids=quizes,
    solutions=solutions,
    solve_grid=lambda grid: solve_grid(
        grid=grid,
        guess_strategy=smallest_allowed_guess_strategy
    )
)

measure_time(
    label="neural",
    num_executions=num_executions,
    grids=quizes,
    solutions=solutions,
    solve_grid=lambda grid: solve_grid(
        grid=grid,
        guess_strategy=neural_guess_strategy
    )
)

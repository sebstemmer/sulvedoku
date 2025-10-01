import time
from typing import Callable
import numpy as np

from grid.grid import Grid, is_equal, str_to_grid
from neural_solve.model import SudokuNet
from solve.solve import solve_grid, random_guess_strategy, ordered_guess_strategy, smallest_allowed_guess_strategy

import torch
from typing import Optional
# from neural_solver.solve import solve
# from neural_solver.fast_neural_solve import fast_neural_solve
import sys

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

sys.exit(0)

sudoku_net = SudokuNet(
    feature_maps=456,
    row_col_box_out_features=27,
    dense_out_features=729
)

sudoku_net.eval()

sudoku_net.load_state_dict(torch.load("/data/neural_solver_weights/97.pth"))


def find_next_coord_and_value_for_neural(
        some_grid: Grid,
        solution: Grid,
        model
) -> Optional[tuple[grid.Coord, int]]:
    start_find_neural = time.perf_counter()

    coord_to_value: dict[grid.Coord, int] = {}
    allowed_values: dict[grid.Coord, list[int]] = {}

    has_0 = False
    for coord in grid.all_coords_0_to_80:
        value = some_grid.cells[coord].value
        coord_to_value[coord] = value

        if value == 0:
            has_0 = True

        allowed_values[coord] = some_grid.cells[coord].allowed_values

    # print(grid.coord_to_str_to_str(lambda x: str(quiz[x]), "\n", " "))
    # print(grid.coord_to_str_to_str(lambda x: str(allowed_values[x]), "\n", " "))
    if has_0:
        a = solve(
            quiz=coord_to_value,
            allowed_values=allowed_values,
            solution=solution,
            model=model
        )

        end_find_neural = time.perf_counter()
        elapsed_ms = (end_find_neural - start_find_neural) * 1000
        # print(f"{elapsed_ms:.3f} ms")
        # print(a)
        return a[0], a[1]
    else:
        return None


sys.exit()

measure_time(
    label="neural",
    num_executions=num_executions,
    grids=quizes,
    solutions=solutions,
    solve_grid=lambda quiz, sol: solve_grid(
        grid=quiz,
        guess_strategy=lambda g: find_next_coord_and_value_for_neural(
            some_grid=g,
            solution=sol,
            model=sudoku_net
        )
    )
)

measure_time(
    label="neural-with-pool",
    num_executions=num_executions,
    grids=quizes,
    solutions=solutions,
    solve_grid=lambda quiz, sol: fast_neural_solve(
        quiz=quiz,
        solution=sol,
        model=sudoku_net
    )
)

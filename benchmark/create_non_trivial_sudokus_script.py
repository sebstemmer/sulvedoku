import csv
from common.track_progress import track
from grid.grid import str_to_grid, grid_to_str, is_equal
from solve.solve import solve_grid_until_no_trivial_solutions

one_million_sudokus_path = "./data/1m-sudokus.csv"
non_trivial_path = "./data/327-non-trivial-sudokus.csv"

with (open(non_trivial_path, "w", newline="", encoding="utf-8") as non_trivial_csv):
    writer = csv.writer(non_trivial_csv, delimiter=",")
    writer.writerow(["idx", "quiz", "solution"])

    csv_lines: list[str] = open(one_million_sudokus_path, 'r').read().splitlines()[1:]

    for idx, csv_line in enumerate(csv_lines):
        track(
            idx=idx,
            total=len(csv_lines),
            output_every=1000,
        )

        grid_str, solution_str = csv_line.split(",")

        grid = str_to_grid(grid_str)
        solution = str_to_grid(solution_str)

        solved_until_no_trivial = solve_grid_until_no_trivial_solutions(grid)

        if not is_equal(grid1=solution, grid2=solved_until_no_trivial.grid):
            print(idx)
            writer.writerow([
                idx,
                grid_to_str(
                    grid=solved_until_no_trivial.grid,
                    row_join="",
                    col_join=""
                ),
                solution_str
            ])

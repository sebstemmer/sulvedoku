import csv
import grid
import dynamic_path

sudokus_path = "./data/1m-sudokus.csv"
non_trivial_path = "./data/non-trivial-sudokus.csv"

with open(non_trivial_path, "w", newline="", encoding="utf-8") as non_trivial_csv:
    writer = csv.writer(non_trivial_csv, delimiter=",")
    writer.writerow(["idx", "quiz", "solution"])

    for idx, csv_line in enumerate(open(sudokus_path, 'r').read().splitlines()[1:]):
        if idx % 10000 == 0:
            print(idx)

        quiz_str, solution_str = csv_line.split(",")

        quiz = grid.str_to_grid(quiz_str)

        solution = dynamic_path.solve_valid_grid_until_no_trivial_solutions(quiz)

        if grid.grid_to_str(
                grid=solution.grid,
                row_join="",
                col_join=""
        ) != solution_str:
            print(idx)
            writer.writerow([
                idx,
                grid.grid_to_str(
                    grid=solution.grid,
                    row_join="",
                    col_join=""
                ),
                solution_str
            ])

from grid.grid import str_to_grid, all_coords_0_to_80
import numpy as np
import csv
from common.track_progress import track
from solve.solve import solve_grid_until_no_trivial_solutions
from neural_solve.train_utils import train_data_path

raw_data_path = "./data/3m-sudokus.csv"

def get_train_data_row(
        quiz_str: str,
        solution_str: str,
        difficulty_str: str
) -> tuple[str, str, str] | None:
    if float(difficulty_str) < 0.1:
        return None

    quiz_str: str = quiz_str.replace(".", "0")

    solved_until_no_trivial_solutions = solve_grid_until_no_trivial_solutions(
        grid=str_to_grid(quiz_str)
    )

    if len(solved_until_no_trivial_solutions.grid.filled_coords) == 0:
        return None

    solution_chars = list(solution_str)

    input_str: str = ""
    target_str: str = ""
    mask_str: str = ""
    for coord in all_coords_0_to_80:
        cell = solved_until_no_trivial_solutions.grid.cells[coord]

        value_one_hot = np.zeros(10, dtype=np.uint8)
        value: int = cell.value
        value_one_hot[value] = 1
        value_one_hot_str: str = "".join(str(v) for v in value_one_hot)

        allowed_values = np.array(cell.allowed_values, dtype=np.uint8) - 1
        allowed_values_one_hot = np.zeros(9, dtype=np.uint8)
        allowed_values_one_hot[allowed_values] = 1
        allowed_values_one_hot_str: str = "".join(str(a) for a in allowed_values_one_hot)

        input_str += value_one_hot_str + allowed_values_one_hot_str

        solution_value_indexed: int = int(solution_chars[coord.entry_idx]) - 1

        target_str += str(solution_value_indexed)

        mask_str += ("1") if value == 0 else ("0")

    return (input_str, target_str, mask_str)


with open(train_data_path, "w", newline="", encoding="utf-8") as f_csv:
    writer = csv.writer(f_csv, delimiter=",")
    writer.writerow(["input", "target", "mask"])

    all_lines: list[str] = open(raw_data_path, 'r').read().splitlines()[1:]
    for line_idx, line in enumerate(all_lines):
        track(
            idx=line_idx,
            total=len(all_lines),
            output_every=1000,
        )

        _, quiz_str, solution_str, _, difficulty_str = line.split(",")

        train_data_row: tuple[str, str, str] = get_train_data_row(
            quiz_str=quiz_str,
            solution_str=solution_str,
            difficulty_str=difficulty_str
        )

        if train_data_row is not None:
            input_str, target_str, mask_str = train_data_row

            writer.writerow([input_str, target_str, mask_str])

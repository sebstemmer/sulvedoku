import csv
import grid
import dynamic_path
import time
from track_progress import track

filled_path = "./data/filled_2.csv"

num_filled: int = 900_000
start = time.perf_counter()

with open(filled_path, "w", newline="", encoding="utf-8") as filled_csv:
    writer = csv.writer(filled_csv, delimiter=",")

    for i in range(num_filled):
        track(
            idx=i,
            total=num_filled,
            output_every=1000,
            counter=start
        )

        filled = dynamic_path.create_filled(
            max_go_back_depth=-1,
            method=dynamic_path.FillPathCreationMethod.RANDOM
        )

        writer.writerow([
            grid.grid_to_str(
                grid=filled.grid,
                row_join="",
                col_join=""
            )
        ])

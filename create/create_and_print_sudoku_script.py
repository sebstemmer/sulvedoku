from create.create import create_grid
from grid.grid import Coord, get_entry_idx

num_filled_target: int = 23


result = create_grid(
    num_filled_target=num_filled_target,
    max_remove_depth=100
)

total = ""
for row in range(0, 9):
    for col in range(0, 9):
        value = result.cells[Coord(row, col, get_entry_idx(row, col))].value

        if value == 0:
            value_str = "_"
        else:
            value_str = str(value)

        if col == 2 or col == 5:
            total += value_str + "   |   "
        elif col < 8:
            total += value_str + "    "
        else:
            total += value_str

    if row == 2 or row == 5:
        total += str("\n\n------------------------------------------------\n\n")
    elif row < 8:
        total += str("\n\n")
    else:
        pass

print(total)

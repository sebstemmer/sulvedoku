from typing import TypeVar
import random

T = TypeVar('T')


def flatten(list_of_lists: list[list[T]]) -> list[T]:
    return [item for sublist in list_of_lists for item in sublist]


def get_all_in_col(
        col_idx: int,
        grid: list[list[int]],
) -> list[int]:
    return [grid[row_idx][col_idx] for row_idx in range(0, 9)]


def get_all_in_row(
        row_idx: int,
        grid: list[list[int]],
) -> list[int]:
    return [grid[row_idx][col_idx] for col_idx in range(0, 9)]


def get_all_in_square(
        row_start_idx: int,
        col_start_idx: int,
        grid: list[list[int]],
) -> list[int]:
    return flatten([
        [
            grid[row_idx][col_idx] for col_idx in range(
                col_start_idx,
                col_start_idx + 3
            )
        ] for row_idx in range(
            row_start_idx,
            row_start_idx + 3
        )
    ])


def get_allowed_values(
        row_idx: int,
        col_idx: int,
        grid: list[list[int]],
        row_col_idx_to_square: dict[tuple[int, int], int],
        square_idx_to_row_col_start_idx: dict[int, tuple[int, int]]
) -> list[int]:
    in_row: list[int] = get_all_in_row(
        row_idx=row_idx,
        grid=grid
    )
    in_col: list[int] = get_all_in_col(
        col_idx=col_idx,
        grid=grid
    )

    square_idx: int = row_col_idx_to_square[(row_idx, col_idx)]

    row_col_start_idx: tuple[int,
                             int] = square_idx_to_row_col_start_idx[square_idx]

    in_square: list[int] = get_all_in_square(
        row_start_idx=row_col_start_idx[0],
        col_start_idx=row_col_start_idx[1],
        grid=grid
    )

    all: set[int] = set(in_row + in_col + in_square)

    allowed: list[int] = [n for n in range(1, 10) if n not in all]

    return allowed


def get_square_idx(row_idx: int, col_idx: int) -> int:
    if row_idx >= 0 and row_idx < 3 and col_idx >= 0 and col_idx < 3:
        return 0
    if row_idx >= 0 and row_idx < 3 and col_idx >= 3 and col_idx < 6:
        return 1
    if row_idx >= 0 and row_idx < 3 and col_idx >= 6 and col_idx < 9:
        return 2
    if row_idx >= 3 and row_idx < 6 and col_idx >= 0 and col_idx < 3:
        return 3
    if row_idx >= 3 and row_idx < 6 and col_idx >= 3 and col_idx < 6:
        return 4
    if row_idx >= 3 and row_idx < 6 and col_idx >= 6 and col_idx < 9:
        return 5
    if row_idx >= 6 and row_idx < 9 and col_idx >= 0 and col_idx < 3:
        return 6
    if row_idx >= 6 and row_idx < 9 and col_idx >= 3 and col_idx < 6:
        return 7
    if row_idx >= 6 and row_idx < 9 and col_idx >= 6 and col_idx < 9:
        return 8
    raise RuntimeError(
        "indices out of bounds, row_idx: " +
        str(row_idx) + " , col_idx: " + str(col_idx)
    )


def create_row_col_idx_to_square_idx() -> dict[tuple[int, int], int]:
    row_col_idx_to_square_idx: dict[tuple[int, int], int] = {}

    for row_idx in range(0, 9):
        for col_idx in range(0, 9):
            row_col_idx_to_square_idx[(row_idx, col_idx)] = get_square_idx(
                row_idx=row_idx,
                col_idx=col_idx
            )

    return row_col_idx_to_square_idx


def create_square_idx_to_row_col_start_idx() -> dict[int, tuple[int, int]]:
    return {
        0: (0, 0),
        1: (0, 3),
        2: (0, 6),
        3: (3, 0),
        4: (3, 3),
        5: (3, 6),
        6: (6, 0),
        7: (6, 3),
        8: (6, 6)
    }


def init_grid() -> list[list[int]]:
    return [
        [0 for _ in range(0, 9)] for _ in range(0, 9)
    ]


def make_guess(
        allowed_values: list[int],
) -> int:
    idx_to_allowed: dict[int, int] = {
        idx: value for idx, value in enumerate(allowed_values)
    }
    guess_idx: int = random.randint(0, len(allowed_values) - 1)

    return idx_to_allowed[guess_idx]


class GridInvalid(Exception):
    def __init__(self, row_idx: int, col_idx: int):
        msg: str = "row_idx: " + str(row_idx) + ", col_idx: " + str(col_idx)
        super().__init__(msg)


def handle_cell(
    row_idx: int,
    col_idx: int,
    grid: list[list[int]],
    row_col_idx_to_square: dict[tuple[int, int], int],
    square_idx_to_row_col_start_idx: dict[int, tuple[int, int]]
) -> list[list[int]]:
    allowed_values: list[int] = get_allowed_values(
        row_idx=row_idx,
        col_idx=col_idx,
        grid=grid,
        row_col_idx_to_square=row_col_idx_to_square,
        square_idx_to_row_col_start_idx=square_idx_to_row_col_start_idx
    )

    if len(allowed_values) == 0:
        raise GridInvalid(row_idx, col_idx)

    if len(allowed_values) == 1:
        grid[row_idx][col_idx] = allowed_values[0]
        return grid

    guessed: int = make_guess(
        allowed_values=allowed_values
    )

    grid[row_idx][col_idx] = guessed

    return grid


def create_grid(
    row_col_idx_to_square_idx: dict[tuple[int, int], int],
    square_idx_to_row_col_start_idx: dict[int, tuple[int, int]]
) -> list[list[int]]:
    grid: list[list[int]] = init_grid()

    for row_idx in range(0, 9):
        for col_idx in range(0, 9):
            grid: list[list[int]] = handle_cell(
                row_idx=row_idx,
                col_idx=col_idx,
                grid=grid,
                row_col_idx_to_square=row_col_idx_to_square_idx,
                square_idx_to_row_col_start_idx=square_idx_to_row_col_start_idx
            )

    return grid


def recursive_create_grid(
    row_col_idx_to_square_idx: dict[tuple[int, int], int],
    square_idx_to_row_col_start_idx: dict[int, tuple[int, int]]
) -> list[list[int]]:
    max_num_tries: int = 10000000
    for create_try in range(0, max_num_tries):
        try:
            grid: list[list[int]] = create_grid(
                row_col_idx_to_square_idx=row_col_idx_to_square_idx,
                square_idx_to_row_col_start_idx=square_idx_to_row_col_start_idx
            )
            print("needed " + str(create_try) + " create tries")
            return grid
        except GridInvalid:
            pass

    raise RuntimeError(str(max_num_tries) + " where not enough")


row_col_idx_to_square_idx: dict[
    tuple[int, int], int
] = create_row_col_idx_to_square_idx()

square_idx_to_row_col_start_idx: dict[
    int, tuple[int, int]
] = create_square_idx_to_row_col_start_idx()

grid: list[list[int]] = recursive_create_grid(
    row_col_idx_to_square_idx=row_col_idx_to_square_idx,
    square_idx_to_row_col_start_idx=square_idx_to_row_col_start_idx
)

print(grid)

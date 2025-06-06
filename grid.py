from typing import NamedTuple, Optional
import random

# todo: use frozenset


class Coord(NamedTuple):
    row_idx: int
    col_idx: int


def create_all_coords() -> list[Coord]:
    return [
        Coord(
            row_idx=row_idx,
            col_idx=col_idx
        )
        for col_idx in range(0, 9)
        for row_idx in range(0, 9)
    ]


def get_coords_in_square(square_idx: int) -> list[Coord]:
    if not (0 <= square_idx <= 8):
        raise ValueError(f"{square_idx} must be between 0 and 8")

    square_row_idx = square_idx // 3
    square_col_idx = square_idx % 3

    return [
        Coord(
            row_idx=row_idx,
            col_idx=col_idx
        ) for row_idx in range(
            3 * square_row_idx,
            3 * square_row_idx + 3
        ) for col_idx in range(
            3 * square_col_idx,
            3 * square_col_idx + 3
        )
    ]


def get_coords_in_square_of_coord(
    coord: Coord


) -> list[Coord]:
    for square_idx in range(0, 9):
        coords_in_square: list[Coord] = get_coords_in_square(
            square_idx=square_idx
        )
        if coord in coords_in_square:
            return coords_in_square

    raise ValueError(
        f"square_idx not found for {coord}"
    )


def create_coord_to_all_coords_in_row_col_or_square() -> dict[Coord, set[Coord]]:
    coord_to_all_coords_in_row_col_or_square: dict[
        Coord, set[Coord]
    ] = {}

    for coord in create_all_coords():
        in_row: list[Coord] = [
            Coord(
                row_idx=coord.row_idx,
                col_idx=col_idx
            ) for col_idx in range(0, 9)
        ]

        in_col: list[Coord] = [
            Coord(
                row_idx=row_idx,
                col_idx=coord.col_idx
            ) for row_idx in range(0, 9)
        ]

        in_square: list[Coord] = get_coords_in_square_of_coord(
            coord=coord
        )

        in_row_col_or_square: set[Coord] = set(
            in_row + in_col + in_square)

        in_row_col_or_square_without_coord: set[Coord] = {
            c for c in in_row_col_or_square if c != coord
        }

        coord_to_all_coords_in_row_col_or_square[coord] = in_row_col_or_square_without_coord

    return coord_to_all_coords_in_row_col_or_square


class Cell(NamedTuple):
    value: int
    allowed_values: list[int]


def copy_cell_with_new_allowed_values(
    cell: Cell,
    new_allowed_values: list[int]
) -> Cell:
    return Cell(
        value=cell.value,
        allowed_values=new_allowed_values
    )


def copy_cell(
    cell: Cell,
) -> Cell:
    return copy_cell_with_new_allowed_values(
        cell=cell,
        new_allowed_values=cell.allowed_values
    )


def copy_grid(
    grid: dict[Coord, Cell]
) -> dict[Coord, Cell]:
    return {
        coord_to_cell[0]: copy_cell(coord_to_cell[1]) for coord_to_cell in grid.items()
    }


def create_empty_grid(
    all_coords: list[Coord],
) -> dict[Coord, Cell]:
    return {
        coord: Cell(
            value=0,
            allowed_values=list(range(1, 10))
        ) for coord in all_coords
    }


def set_value_in_grid(
    grid: dict[Coord, Cell],
    coord: Coord,
    value: int,
    coord_to_all_coords_in_row_col_or_square: dict[
        Coord,
        set[Coord]
    ]
) -> dict[Coord, Cell]:
    new_grid: dict[Coord, Cell] = copy_grid(grid)

    new_grid[coord] = Cell(
        value=value,
        allowed_values=[]
    )

    coords_in_row_col_or_square: set[Coord] = coord_to_all_coords_in_row_col_or_square[
        coord
    ]

    for coord_with_changed_allowed_values in coords_in_row_col_or_square:
        old_cell: Cell = grid[coord_with_changed_allowed_values]
        new_grid[coord_with_changed_allowed_values] = copy_cell_with_new_allowed_values(
            cell=old_cell,
            new_allowed_values=[
                v for v in old_cell.allowed_values if v != value
            ]
        )

    return new_grid


def remove_values_from_grid(
        grid: dict[Coord, Cell],
        coords: list[Coord],
        coord_to_all_coords_in_row_col_or_square: dict[
            Coord,
            set[Coord]
        ]
) -> dict[Coord, Cell]:
    new_grid: dict[Coord, Cell] = copy_grid(grid)

    for coord in coords:
        new_grid[coord] = Cell(
            value=0,
            allowed_values=[]
        )

    all_affected_coords: set[Coord] = set()

    for coord in coords:
        affected_coords: set[Coord] = coord_to_all_coords_in_row_col_or_square[coord]

        all_affected_coords |= affected_coords

    all_affected_coords |= set(coords)

    for affected_coord in all_affected_coords:
        affected_cell: Cell = new_grid[affected_coord]

        if affected_cell.value == 0:
            already_used: set[int] = set(
                [
                    new_grid[constrain_coord].value for constrain_coord in coord_to_all_coords_in_row_col_or_square[affected_coord]
                ]
            )
            new_grid[affected_coord] = copy_cell_with_new_allowed_values(
                cell=new_grid[affected_coord],
                new_allowed_values=[v for v in range(
                    1, 10
                ) if v not in already_used]
            )

    return new_grid


def get_random_empty_where_allowed_values_is_len_1(
    grid: dict[Coord, Cell]
) -> Optional[tuple[Coord, Cell]]:
    new_grid: dict[Coord, Cell] = copy_grid(grid)

    coord_to_cells: list[tuple[Coord, Cell]] = list(
        new_grid.items()
    )

    random.shuffle(coord_to_cells)

    for coord_to_cell in coord_to_cells:
        if len(coord_to_cell[1].allowed_values) == 1 and coord_to_cell[1].value == 0:
            return (
                coord_to_cell[0],
                coord_to_cell[1]
            )

    return None


def grid_to_str(
        grid: dict[Coord, Cell],
        row_join: str,
        col_join: str
) -> str:
    return row_join.join([col_join.join(
        [
            str(grid[Coord(row_idx=row_idx, col_idx=col_idx)].value) for col_idx in range(0, 9)
        ]) for row_idx in range(0, 9)])


def is_valid(
    grid: dict[Coord, Cell],
    coord_to_all_coords_in_row_col_or_square: dict[
        Coord, set[Coord]
    ]
) -> bool:
    for coord in coord_to_all_coords_in_row_col_or_square.keys():
        values: set[int] = {
            grid[c].value for c in coord_to_all_coords_in_row_col_or_square[coord]
        }
        value = grid[coord].value

        if value == 0 and len(values) == 0:
            return False

        if value > 0 and value in values:
            return False

    return True


def str_to_grid(
        grid_as_str: str,
        coord_to_all_coords_in_row_col_or_square: dict[
            Coord, set[Coord]
        ]) -> dict[Coord, Cell]:
    if (len(grid_as_str) != 81):
        raise ValueError("size must be 81")

    grid = create_empty_grid(
        all_coords=list(
            coord_to_all_coords_in_row_col_or_square.keys()
        )
    )

    for idx, char in enumerate(grid_as_str):
        row_idx = idx // 9
        col_idx = idx - row_idx * 9

        value = int(char)

        if value > 0:
            grid = set_value_in_grid(
                grid=grid,
                coord=Coord(
                    row_idx=row_idx,
                    col_idx=col_idx
                ),
                value=value,
                coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
            )

    return grid

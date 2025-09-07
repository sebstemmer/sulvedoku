from typing import NamedTuple, Optional, Callable
from frozendict import frozendict
import random


class Coord(NamedTuple):
    row_idx: int
    col_idx: int


class Cell(NamedTuple):
    value: int
    allowed_values: tuple[int, ...]


class Grid(NamedTuple):
    cells: dict[Coord, Cell]
    empty_coords: tuple[Coord, ...]


def create_all_coords() -> list[Coord]:
    coords: list[Coord] = []

    for entry_idx in range(0, 81):
        row_idx: int = entry_idx // 9
        col_idx: int = entry_idx - row_idx * 9

        coords.append(Coord(
            row_idx=row_idx,
            col_idx=col_idx
        ))

    return coords


all_coords_0_to_80: tuple[Coord, ...] = tuple(create_all_coords())


def get_coords_in_block(block_idx: int) -> list[Coord]:
    if not (0 <= block_idx <= 8):
        raise ValueError(f"{block_idx} must be between 0 and 8")

    square_row_idx = block_idx // 3
    square_col_idx = block_idx % 3

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


def get_coords_in_block_of_coord(
        coord: Coord
) -> list[Coord]:
    for block_idx in range(0, 9):
        coords_in_square: list[Coord] = get_coords_in_block(
            block_idx=block_idx
        )
        if coord in coords_in_square:
            return coords_in_square

    raise ValueError(
        f"block_idx not found for {coord}"
    )


def create_coord_to_all_coords_in_row_col_or_block() -> frozendict[Coord, set[Coord]]:
    coord_to_all_coords_in_row_col_or_block: dict[
        Coord, set[Coord]
    ] = {}

    for coord in all_coords_0_to_80:
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

        in_block: list[Coord] = get_coords_in_block_of_coord(
            coord=coord
        )

        in_row_col_or_block: set[Coord] = set(
            in_row + in_col + in_block
        )

        in_row_col_or_block_without_coord: set[Coord] = {
            c for c in in_row_col_or_block if c != coord
        }

        coord_to_all_coords_in_row_col_or_block[coord] = in_row_col_or_block_without_coord

    return frozendict(coord_to_all_coords_in_row_col_or_block)


coord_to_all_coords_in_row_col_or_block = create_coord_to_all_coords_in_row_col_or_block()


def copy_grid(
        grid: Grid
) -> Grid:
    return Grid(
        cells={
            coord_to_cell[0]: Cell(*coord_to_cell[1]) for coord_to_cell in grid.cells.items()
        },
        empty_coords=grid.empty_coords
    )


def create_empty_grid() -> Grid:
    return Grid(
        cells={coord: Cell(
            value=0,
            allowed_values=tuple(range(1, 10))
        ) for coord in all_coords_0_to_80
        },
        empty_coords=all_coords_0_to_80
    )


def set_value_in_grid(
        grid: Grid,
        coord: Coord,
        value: int,
) -> Grid | None:
    news_cells: dict[Coord, Cell] = grid.cells.copy()

    news_cells[coord] = Cell(
        value=value,
        allowed_values=()
    )

    coords_in_row_col_or_square: set[Coord] = coord_to_all_coords_in_row_col_or_block[
        coord
    ]

    for coord_with_changed_allowed_values in coords_in_row_col_or_square:
        old_cell: Cell = grid.cells[coord_with_changed_allowed_values]
        if old_cell.value == 0:
            new_allowed_values: tuple[int, ...] = tuple([v for v in old_cell.allowed_values if v != value])

            len_new_allowed_values: int = len(new_allowed_values)

            if len_new_allowed_values == 0:
                return None

            news_cells[coord_with_changed_allowed_values] = Cell(
                value=old_cell.value,
                allowed_values=new_allowed_values
            )

    return Grid(
        cells=news_cells,
        empty_coords=tuple([c for c in grid.empty_coords if c != coord])
    )


def remove_values_from_grid(
        grid: Grid,
        coords: list[Coord]
) -> Grid:
    new_cells: dict[Coord, Cell] = grid.cells.copy()

    for coord in coords:
        new_cells[coord] = Cell(
            value=0,
            allowed_values=()
        )

    all_affected_coords: set[Coord] = set()

    for coord in coords:
        affected_coords: set[Coord] = coord_to_all_coords_in_row_col_or_block[coord]

        all_affected_coords |= affected_coords

    all_affected_coords |= set(coords)

    total_is_valid: bool = grid.is_valid
    for affected_coord in all_affected_coords:
        affected_cell: Cell = new_cells[affected_coord]

        if affected_cell.value == 0:
            already_used: set[int] = set(
                [
                    new_cells[constrain_coord].value for constrain_coord in
                    coord_to_all_coords_in_row_col_or_block[affected_coord]
                ]
            )
            new_allowed_values: tuple[int, ...] = tuple([v for v in range(
                1, 10
            ) if v not in already_used])

            total_is_valid = total_is_valid and len(new_allowed_values) > 0

            new_cells[affected_coord] = Cell(
                value=new_cells[affected_coord].value,
                allowed_values=new_allowed_values
            )

    return Grid(
        cells=new_cells,
        empty_coords=tuple([c for c in grid.empty_coords if c not in coords])
    )


def get_random_empty_where_allowed_values_is_len_1(
        grid: Grid
) -> Optional[tuple[Coord, Cell]]:
    new_cells: dict[Coord, Cell] = grid.cells.copy()

    coord_to_cells: list[tuple[Coord, Cell]] = list(
        new_cells.items()
    )

    random.shuffle(coord_to_cells)

    for coord_to_cell in coord_to_cells:
        if len(coord_to_cell[1].allowed_values) == 1 and coord_to_cell[1].value == 0:
            return (
                coord_to_cell[0],
                coord_to_cell[1]
            )

    return None


def coord_to_str_to_str(
        coord_to_str: Callable[[Coord], str],
        row_join: str,
        col_join: str
) -> str:
    return row_join.join([col_join.join(
        [
            coord_to_str(Coord(row_idx=row_idx, col_idx=col_idx)) for col_idx in range(0, 9)
        ]) for row_idx in range(0, 9)])


def grid_to_str(
        grid: Grid,
        row_join: str,
        col_join: str
) -> str:
    return coord_to_str_to_str(
        coord_to_str=lambda c: str(grid.cells[c].value),
        row_join=row_join,
        col_join=col_join
    )


def str_to_grid(
        grid_as_str: str
) -> Grid:
    if len(grid_as_str) != 81:
        raise ValueError("size must be 81")

    grid: Grid = create_empty_grid()

    for idx, char in enumerate(grid_as_str):
        row_idx: int = idx // 9
        col_idx: int = idx - row_idx * 9

        value = int(char)

        if value < 0 or value > 9:
            raise ValueError(f"{value} must be between 0 (incl) and 9 (incl)")

        if value > 0:
            grid = set_value_in_grid(
                grid=grid,
                coord=Coord(
                    row_idx=row_idx,
                    col_idx=col_idx
                ),
                value=value
            )

    return grid


def is_equal(grid1: Grid, grid2: Grid) -> bool:
    return grid1.cells == grid2.cells

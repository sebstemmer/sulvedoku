from typing import NamedTuple, Callable

from frozendict import frozendict


class Coord(NamedTuple):
    """
        Coordinate in a sudoku.

        Attributes:
            row_idx (int):
                Row index from 0 (incl) to 8 (incl).
            col_idx (int):
                Column index from 0 (incl) to 8 (incl).
    """
    row_idx: int
    col_idx: int
    entry_idx: int


class Cell(NamedTuple):
    """
        Cell in a sudoku.

        Attributes:
            value (int):
                0 (cell is empty) or range from 1 (incl) to 9 (incl).
            allowed_values (tuple[int, ...]):
                Column, row and block constrain allowed values in a cell. Range from 1 (incl) to 9 (incl))
    """
    value: int
    allowed_values: tuple[int, ...]


# todo : unsortiert besser, lieber in guess strategy ordered sortieren + choicoe verwencen
class Grid(NamedTuple):
    """
        Sudoku.

        Attributes:
            cells (frozendict[Coord, Cell]):
                All cells.
            empty_coords (dict[Coord, bool]):
                Coordinates where cells are empty (value == 0), unsorted.
            filled_coords (tuple[Coord, ...]):
                Coordinates where cells are filled (value != 0), unsorted.
    """
    cells: frozendict[Coord, Cell]
    empty_coords: tuple[Coord, ...]
    filled_coords: tuple[Coord, ...]


def get_entry_idx(row_idx: int, col_idx: int) -> int:
    """
        Get entry index from row index and column index in row format.

        Args:
            row_idx (int):
                Row index of coordinate to get entry index.
            col_idx (int):
                Column index of coordinate to get entry index.

        Returns:
            int:
                Entry index from 0 (incl) to 80 (incl).
    """
    return row_idx * 9 + col_idx


def get_coord(entry_idx: int) -> Coord:
    """
        Get coordinate from row format entry index.

        Args:
            entry_idx (int):
                Entry index of coordinate.

        Returns:
            Coord:
               Coordinate to entry index.
    """

    row_idx: int = entry_idx // 9

    return Coord(
        row_idx=row_idx,
        col_idx=entry_idx - row_idx * 9,
        entry_idx=entry_idx
    )


def create_all_coords() -> list[Coord]:
    """
        Creates all possible coordinates in a sudoku.

        Returns:
            list[Coord]:
                All 81 coordinates in row format.
    """
    coords: list[Coord] = []

    for entry_idx in range(0, 81):
        row_idx: int = entry_idx // 9
        col_idx: int = entry_idx - row_idx * 9

        coords.append(Coord(
            row_idx=row_idx,
            col_idx=col_idx,
            entry_idx=entry_idx
        ))

    return coords


all_coords_0_to_80: tuple[Coord, ...] = tuple(create_all_coords())


def get_coords_in_box(box_idx: int) -> list[Coord]:
    """
        Get all coordinates in box.

        Args:
            box_idx (int):
                Index of box in row format.

        Returns:
            list[Coord]:
                All coordinates in box.
    """
    if not (0 <= box_idx <= 8):
        raise ValueError(f"{box_idx} must be between 0 and 8")

    square_row_idx = box_idx // 3
    square_col_idx = box_idx % 3

    return [
        Coord(
            row_idx=row_idx,
            col_idx=col_idx,
            entry_idx=get_entry_idx(
                row_idx=row_idx,
                col_idx=col_idx
            )
        ) for row_idx in range(
            3 * square_row_idx,
            3 * square_row_idx + 3
        ) for col_idx in range(
            3 * square_col_idx,
            3 * square_col_idx + 3
        )
    ]


def get_coords_in_box_of_coord(
        coord: Coord
) -> list[Coord]:
    """
        Get all coordinates in box where coord is in.

        Args:
            coord (Coord):
                Coordinate for which to get box coordinates.

        Returns:
            list[Coord]:
                All coordinates in box where coord is in.
    """
    for block_idx in range(0, 9):
        coords_in_square: list[Coord] = get_coords_in_box(
            box_idx=block_idx
        )
        if coord in coords_in_square:
            return coords_in_square

    raise ValueError(
        f"block_idx not found for {coord}"
    )


def create_coord_to_all_coords_in_row_col_or_block() -> frozendict[Coord, set[Coord]]:
    """
        Get a map from all coordinates in grid to all coordinates that constrain
        this coordinate (excluding the coordinate under consideration).

        Returns:
            frozendict[Coord, set[Coord]]: Coordinate to set of coordinates constraining it (excluding itself).
    """
    coord_to_all_coords_in_row_col_or_block: dict[
        Coord, set[Coord]
    ] = {}

    for coord in all_coords_0_to_80:
        in_row: list[Coord] = [
            Coord(
                row_idx=coord.row_idx,
                col_idx=col_idx,
                entry_idx=get_entry_idx(
                    row_idx=coord.row_idx,
                    col_idx=col_idx
                )
            ) for col_idx in range(0, 9)
        ]

        in_col: list[Coord] = [
            Coord(
                row_idx=row_idx,
                col_idx=coord.col_idx,
                entry_idx=get_entry_idx(
                    row_idx=row_idx,
                    col_idx=coord.col_idx
                )
            ) for row_idx in range(0, 9)
        ]

        in_block: list[Coord] = get_coords_in_box_of_coord(
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


def create_empty_grid() -> Grid:
    """
        Create an empty grid.

        Returns:
            Grid: Empty grid.
    """
    return Grid(
        cells=frozendict({coord: Cell(
            value=0,
            allowed_values=tuple(range(1, 10))
        ) for coord in all_coords_0_to_80
        }),
        empty_coords=all_coords_0_to_80,
        filled_coords=()
    )


def remove_coord_from_empty_or_filled_coords(
        coord: Coord,
        empty_or_filled_coords: tuple[Coord, ...]
) -> tuple[Coord, ...]:
    coord_idx = empty_or_filled_coords.index(coord)
    return empty_or_filled_coords[:coord_idx] + empty_or_filled_coords[coord_idx + 1:]


def set_value_in_grid(
        grid: Grid,
        coord: Coord,
        value: int,
) -> Grid | None:
    """
        Create an empty grid.

        Returns:
            Grid: Empty grid.
    """
    news_cells: dict[Coord, Cell] = dict(grid.cells)

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
        cells=frozendict(news_cells),
        empty_coords=remove_coord_from_empty_or_filled_coords(
            coord=coord,
            empty_or_filled_coords=grid.empty_coords
        ),
        filled_coords=grid.filled_coords + (coord,)
    )


def remove_value_from_grid(
        grid: Grid,
        coord: Coord
) -> Grid:
    new_cells: dict[Coord, Cell] = dict(grid.cells)

    new_cells[coord] = Cell(
        value=0,
        allowed_values=()
    )

    affected_coords: set[Coord] = coord_to_all_coords_in_row_col_or_block[coord] | {coord}

    for affected_coord in affected_coords:
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

            new_cells[affected_coord] = Cell(
                value=new_cells[affected_coord].value,
                allowed_values=new_allowed_values
            )

    return Grid(
        cells=frozendict(new_cells),
        empty_coords=grid.empty_coords + (coord,),
        filled_coords=remove_coord_from_empty_or_filled_coords(
            coord=coord,
            empty_or_filled_coords=grid.filled_coords
        )
    )


def coord_to_str_to_str(
        coord_to_str: Callable[[Coord], str],
        row_join: str,
        col_join: str
) -> str:
    return row_join.join([col_join.join(
        [
            coord_to_str(
                Coord(row_idx=row_idx, col_idx=col_idx, entry_idx=get_entry_idx(row_idx=row_idx, col_idx=col_idx))) for
            col_idx in range(0, 9)
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

    for entry_idx, char in enumerate(grid_as_str):
        value = int(char) if char in "0123456789" else 0

        if value > 0:
            grid = set_value_in_grid(
                grid=grid,
                coord=get_coord(entry_idx=entry_idx),
                value=value
            )

    return grid


def is_equal(grid1: Grid, grid2: Grid) -> bool:
    return grid1.cells == grid2.cells

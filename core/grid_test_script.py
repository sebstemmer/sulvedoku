from typing import Optional

from grid import all_coords_0_to_80, coord_to_all_coords_in_row_col_or_block, get_coords_in_block, \
    get_coords_in_block_of_coord, copy_grid, set_value_in_grid, \
    create_empty_grid, remove_values_from_grid, get_random_empty_where_allowed_values_is_len_1, Coord, Cell, Grid


def create_all_coords_test() -> None:
    assert all_coords_0_to_80[0] == Coord(row_idx=0, col_idx=0)

    assert all_coords_0_to_80[15] == Coord(row_idx=1, col_idx=6)

    assert len(all_coords_0_to_80) == 81


def get_coords_in_block_test() -> None:
    coords: list[Coord] = get_coords_in_block(block_idx=4)

    assert coords == [
        Coord(row_idx=3, col_idx=3),
        Coord(row_idx=3, col_idx=4),
        Coord(row_idx=3, col_idx=5),
        Coord(row_idx=4, col_idx=3),
        Coord(row_idx=4, col_idx=4),
        Coord(row_idx=4, col_idx=5),
        Coord(row_idx=5, col_idx=3),
        Coord(row_idx=5, col_idx=4),
        Coord(row_idx=5, col_idx=5),
    ]


def get_coords_in_block_of_coord_test() -> None:
    coords: list[Coord] = get_coords_in_block_of_coord(
        Coord(
            row_idx=3,
            col_idx=2
        )
    )

    assert coords == [
        Coord(row_idx=3, col_idx=0),
        Coord(row_idx=3, col_idx=1),
        Coord(row_idx=3, col_idx=2),
        Coord(row_idx=4, col_idx=0),
        Coord(row_idx=4, col_idx=1),
        Coord(row_idx=4, col_idx=2),
        Coord(row_idx=5, col_idx=0),
        Coord(row_idx=5, col_idx=1),
        Coord(row_idx=5, col_idx=2),
    ]


def create_coord_to_all_coords_in_row_col_or_block_test() -> None:
    assert coord_to_all_coords_in_row_col_or_block[
               Coord(
                   row_idx=4,
                   col_idx=2
               )
           ] == {
               Coord(row_idx=4, col_idx=3),
               Coord(row_idx=4, col_idx=4),
               Coord(row_idx=4, col_idx=5),
               Coord(row_idx=4, col_idx=6),
               Coord(row_idx=4, col_idx=7),
               Coord(row_idx=4, col_idx=8),
               Coord(row_idx=3, col_idx=0),
               Coord(row_idx=3, col_idx=1),
               Coord(row_idx=3, col_idx=2),
               Coord(row_idx=4, col_idx=0),
               Coord(row_idx=4, col_idx=1),
               Coord(row_idx=5, col_idx=0),
               Coord(row_idx=5, col_idx=1),
               Coord(row_idx=5, col_idx=2),
               Coord(row_idx=0, col_idx=2),
               Coord(row_idx=1, col_idx=2),
               Coord(row_idx=2, col_idx=2),
               Coord(row_idx=6, col_idx=2),
               Coord(row_idx=7, col_idx=2),
               Coord(row_idx=8, col_idx=2),
           }


def create_empty_grid_test() -> None:
    grid: Grid = create_empty_grid()

    assert len(grid.cells) == 81

    for i in all_coords_0_to_80:
        assert grid.cells[i].value == 0
        assert grid.cells[i].allowed_values == list(range(1, 10))
        assert grid.is_valid is True


def copy_grid_test() -> None:
    grid: Grid = create_empty_grid()

    coord_0: Coord = Coord(row_idx=0, col_idx=2)
    coord_1: Coord = Coord(row_idx=1, col_idx=2)

    set_value_in_grid(grid=grid, coord=coord_0, value=5)
    set_value_in_grid(grid=grid, coord=coord_1, value=4)

    copied_grid: Grid = copy_grid(grid=grid)

    assert copied_grid == grid

    assert copied_grid.is_valid == grid.is_valid

    assert grid.cells[coord_0] is not copied_grid.cells[coord_0]


def set_value_in_grid_test() -> None:
    empty_grid: Grid = create_empty_grid()

    first_value_at: Coord = Coord(
        row_idx=3,
        col_idx=2
    )

    grid_after_first_value: Grid = set_value_in_grid(
        grid=empty_grid,
        coord=first_value_at,
        value=1
    )

    for c in coord_to_all_coords_in_row_col_or_block[first_value_at]:
        if c != first_value_at:
            assert grid_after_first_value.cells[c].allowed_values == list(
                range(2, 10)
            )

    second_value_at: Coord = Coord(
        row_idx=first_value_at.row_idx,
        col_idx=5
    )

    grid_after_second_value: Grid = set_value_in_grid(
        grid=grid_after_first_value,
        coord=second_value_at,
        value=5
    )

    assert grid_after_second_value.cells[second_value_at] is not grid_after_first_value.cells[second_value_at]

    assert grid_after_second_value.cells[Coord(row_idx=first_value_at.row_idx, col_idx=0)].allowed_values == [
        2, 3, 4, 6, 7, 8, 9
    ]
    assert grid_after_second_value.cells[Coord(row_idx=6, col_idx=second_value_at.col_idx)].allowed_values == [
        1, 2, 3, 4, 6, 7, 8, 9
    ]
    assert grid_after_second_value.cells[Coord(row_idx=4, col_idx=4)].allowed_values == [
        1, 2, 3, 4, 6, 7, 8, 9
    ]

    third_value_at: Coord = Coord(
        row_idx=5,
        col_idx=0
    )

    grid_after_third_value: Grid = set_value_in_grid(
        grid=grid_after_second_value,
        coord=third_value_at,
        value=8
    )

    assert grid_after_third_value.cells[Coord(row_idx=4, col_idx=1)].allowed_values == [
        2, 3, 4, 5, 6, 7, 9
    ]


def remove_values_from_grid_test() -> None:
    empty_grid: Grid = create_empty_grid()

    coord_0: Coord = Coord(row_idx=3, col_idx=2)

    grid_0: Grid = set_value_in_grid(
        grid=empty_grid,
        coord=coord_0,
        value=1
    )

    grid_1: Grid = set_value_in_grid(
        grid=grid_0,
        coord=Coord(row_idx=3, col_idx=5),
        value=5
    )

    coord_2: Coord = Coord(row_idx=5, col_idx=0)

    grid_2: Grid = set_value_in_grid(
        grid=grid_1,
        coord=coord_2,
        value=8
    )

    coord_3: Coord = Coord(row_idx=4, col_idx=4)

    grid_3: Grid = set_value_in_grid(
        grid=grid_2,
        coord=coord_3,
        value=4
    )

    after_removal: Grid = remove_values_from_grid(
        grid=grid_3,
        coords=[coord_3, coord_0]
    )

    assert after_removal.cells[coord_0] is not grid_3.cells[coord_0]

    assert after_removal.cells[coord_0].allowed_values == [1, 2, 3, 4, 6, 7, 9]
    assert after_removal.cells[coord_0].value == 0

    assert after_removal.cells[Coord(row_idx=4, col_idx=1)].allowed_values == [
        1, 2, 3, 4, 5, 6, 7, 9
    ]
    assert after_removal.cells[Coord(row_idx=4, col_idx=1)].value == 0

    assert after_removal.cells[Coord(row_idx=3, col_idx=3)].allowed_values == [
        1, 2, 3, 4, 6, 7, 8, 9
    ]
    assert after_removal.cells[Coord(row_idx=3, col_idx=3)].value == 0

    assert after_removal.cells[coord_2].value == 8
    assert after_removal.cells[coord_2].allowed_values == []


def get_random_empty_where_allowed_values_is_len_1_test() -> None:
    some_grid: Grid = Grid(
        cells={
            coord: Cell(
                value=1, allowed_values=[2, 3, 4]
            ) for coord in all_coords_0_to_80
        },
        is_valid=True
    )

    some_grid.cells[Coord(row_idx=0, col_idx=0)] = Cell(
        value=0, allowed_values=[5, 6]
    )

    coord_0: Coord = Coord(row_idx=1, col_idx=2)

    some_grid.cells[coord_0] = Cell(
        value=0, allowed_values=[5]
    )

    coord_1: Coord = Coord(row_idx=3, col_idx=4)

    some_grid.cells[coord_1] = Cell(
        value=0, allowed_values=[4]
    )

    empty_opt: Optional[tuple[Coord, Cell]] = get_random_empty_where_allowed_values_is_len_1(
        grid=some_grid
    )

    assert empty_opt is not None

    assert empty_opt[0] in {
        coord_0,
        coord_1
    }


def is_valid_test() -> None:
    grid: Grid = create_empty_grid()

    for col_idx in range(0, 8):
        grid = set_value_in_grid(
            grid=grid,
            coord=Coord(row_idx=0, col_idx=col_idx),
            value=col_idx + 1
        )

    assert grid.is_valid

    grid = set_value_in_grid(
        grid=grid,
        coord=Coord(row_idx=1, col_idx=8),
        value=9
    )

    assert not grid.is_valid


create_all_coords_test()
get_coords_in_block_test()
get_coords_in_block_of_coord_test()
create_coord_to_all_coords_in_row_col_or_block_test()
copy_grid_test()
create_empty_grid_test()
set_value_in_grid_test()
remove_values_from_grid_test()
get_random_empty_where_allowed_values_is_len_1_test()
is_valid_test()

print("all tests passed")

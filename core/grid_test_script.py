from core.grid import all_coords_0_to_80, coord_to_all_coords_in_row_col_or_block, get_coords_in_box, \
    get_coords_in_box_of_coord, set_value_in_grid, \
    create_empty_grid, Coord, Grid, get_entry_idx, remove_value_from_grid


def create_all_coords_test() -> None:
    assert all_coords_0_to_80[0] == Coord(row_idx=0, col_idx=0, entry_idx=0)

    assert all_coords_0_to_80[15] == Coord(row_idx=1, col_idx=6, entry_idx=15)

    assert len(all_coords_0_to_80) == 81


def get_coords_in_block_test() -> None:
    coords: list[Coord] = get_coords_in_box(box_idx=4)

    assert coords == [
        Coord(row_idx=3, col_idx=3, entry_idx=30),
        Coord(row_idx=3, col_idx=4, entry_idx=31),
        Coord(row_idx=3, col_idx=5, entry_idx=32),
        Coord(row_idx=4, col_idx=3, entry_idx=39),
        Coord(row_idx=4, col_idx=4, entry_idx=40),
        Coord(row_idx=4, col_idx=5, entry_idx=41),
        Coord(row_idx=5, col_idx=3, entry_idx=48),
        Coord(row_idx=5, col_idx=4, entry_idx=49),
        Coord(row_idx=5, col_idx=5, entry_idx=50),
    ]


def get_coords_in_block_of_coord_test() -> None:
    coords: list[Coord] = get_coords_in_box_of_coord(
        Coord(
            row_idx=3,
            col_idx=2,
            entry_idx=29
        )
    )

    assert coords == [
        Coord(row_idx=3, col_idx=0, entry_idx=27),
        Coord(row_idx=3, col_idx=1, entry_idx=28),
        Coord(row_idx=3, col_idx=2, entry_idx=29),
        Coord(row_idx=4, col_idx=0, entry_idx=36),
        Coord(row_idx=4, col_idx=1, entry_idx=37),
        Coord(row_idx=4, col_idx=2, entry_idx=38),
        Coord(row_idx=5, col_idx=0, entry_idx=45),
        Coord(row_idx=5, col_idx=1, entry_idx=46),
        Coord(row_idx=5, col_idx=2, entry_idx=47),
    ]


def create_coord_to_all_coords_in_row_col_or_block_test() -> None:
    assert coord_to_all_coords_in_row_col_or_block[
               Coord(
                   row_idx=4,
                   col_idx=2,
                   entry_idx=38
               )
           ] == {
               Coord(row_idx=4, col_idx=3, entry_idx=39),
               Coord(row_idx=4, col_idx=4, entry_idx=40),
               Coord(row_idx=4, col_idx=5, entry_idx=41),
               Coord(row_idx=4, col_idx=6, entry_idx=42),
               Coord(row_idx=4, col_idx=7, entry_idx=43),
               Coord(row_idx=4, col_idx=8, entry_idx=44),
               Coord(row_idx=3, col_idx=0, entry_idx=27),
               Coord(row_idx=3, col_idx=1, entry_idx=28),
               Coord(row_idx=3, col_idx=2, entry_idx=29),
               Coord(row_idx=4, col_idx=0, entry_idx=36),
               Coord(row_idx=4, col_idx=1, entry_idx=37),
               Coord(row_idx=5, col_idx=0, entry_idx=45),
               Coord(row_idx=5, col_idx=1, entry_idx=46),
               Coord(row_idx=5, col_idx=2, entry_idx=47),
               Coord(row_idx=0, col_idx=2, entry_idx=2),
               Coord(row_idx=1, col_idx=2, entry_idx=11),
               Coord(row_idx=2, col_idx=2, entry_idx=20),
               Coord(row_idx=6, col_idx=2, entry_idx=56),
               Coord(row_idx=7, col_idx=2, entry_idx=65),
               Coord(row_idx=8, col_idx=2, entry_idx=74),
           }


def create_empty_grid_test() -> None:
    grid: Grid = create_empty_grid()

    assert len(grid.cells) == 81

    for i in all_coords_0_to_80:
        assert grid.cells[i].value == 0
        assert grid.cells[i].allowed_values == tuple(range(1, 10))

    assert grid.empty_coords == all_coords_0_to_80
    assert len(grid.filled_coords) == 0


def set_value_in_grid_test() -> None:
    empty_grid: Grid = create_empty_grid()

    first_value_at: Coord = Coord(
        row_idx=3,
        col_idx=2,
        entry_idx=29
    )

    grid_after_first_value: Grid = set_value_in_grid(
        grid=empty_grid,
        coord=first_value_at,
        value=1
    )

    for c in coord_to_all_coords_in_row_col_or_block[first_value_at]:
        if c != first_value_at:
            assert grid_after_first_value.cells[c].allowed_values == tuple(
                range(2, 10)
            )

    second_value_at: Coord = Coord(
        row_idx=first_value_at.row_idx,
        col_idx=5,
        entry_idx=get_entry_idx(row_idx=first_value_at.row_idx, col_idx=5)
    )

    grid_after_second_value: Grid = set_value_in_grid(
        grid=grid_after_first_value,
        coord=second_value_at,
        value=5
    )

    assert grid_after_second_value.cells[second_value_at] is not grid_after_first_value.cells[second_value_at]

    assert grid_after_second_value.cells[Coord(
        row_idx=first_value_at.row_idx,
        col_idx=0,
        entry_idx=get_entry_idx(
            row_idx=first_value_at.row_idx,
            col_idx=0
        ))].allowed_values == tuple([
        2, 3, 4, 6, 7, 8, 9
    ])
    assert grid_after_second_value.cells[Coord(row_idx=6, col_idx=second_value_at.col_idx, entry_idx=get_entry_idx(
        row_idx=6,
        col_idx=second_value_at.col_idx
    ))].allowed_values == tuple([
               1, 2, 3, 4, 6, 7, 8, 9
           ])
    assert grid_after_second_value.cells[Coord(row_idx=4, col_idx=4, entry_idx=40)].allowed_values == tuple([
        1, 2, 3, 4, 6, 7, 8, 9
    ])

    third_value_at: Coord = Coord(
        row_idx=5,
        col_idx=0,
        entry_idx=45
    )

    grid_after_third_value: Grid = set_value_in_grid(
        grid=grid_after_second_value,
        coord=third_value_at,
        value=8
    )

    assert grid_after_third_value.cells[Coord(row_idx=4, col_idx=1, entry_idx=37)].allowed_values == tuple([
        2, 3, 4, 5, 6, 7, 9
    ])


def remove_value_from_grid_test() -> None:
    empty_grid: Grid = create_empty_grid()

    coord_0: Coord = Coord(row_idx=3, col_idx=2, entry_idx=29)

    grid_0: Grid = set_value_in_grid(
        grid=empty_grid,
        coord=coord_0,
        value=1
    )

    grid_1: Grid = set_value_in_grid(
        grid=grid_0,
        coord=Coord(row_idx=3, col_idx=5, entry_idx=32),
        value=5
    )

    coord_2: Coord = Coord(row_idx=5, col_idx=0, entry_idx=45)

    grid_2: Grid = set_value_in_grid(
        grid=grid_1,
        coord=coord_2,
        value=8
    )

    coord_3: Coord = Coord(row_idx=4, col_idx=4, entry_idx=40)

    grid_3: Grid = set_value_in_grid(
        grid=grid_2,
        coord=coord_3,
        value=4
    )

    after_remove_coord_3 = remove_value_from_grid(grid=grid_3, coord=coord_3)

    after_removal: Grid = remove_value_from_grid(
        grid=after_remove_coord_3,
        coord=coord_0
    )

    assert after_removal.cells[coord_0] is not grid_3.cells[coord_0]

    assert after_removal.cells[coord_0].allowed_values == [1, 2, 3, 4, 6, 7, 9]
    assert after_removal.cells[coord_0].value == 0

    assert after_removal.cells[Coord(row_idx=4, col_idx=1, entry_idx=37)].allowed_values == [
        1, 2, 3, 4, 5, 6, 7, 9
    ]
    assert after_removal.cells[Coord(row_idx=4, col_idx=1, entry_idx=37)].value == 0

    assert after_removal.cells[Coord(row_idx=3, col_idx=3, entry_idx=30)].allowed_values == [
        1, 2, 3, 4, 6, 7, 8, 9
    ]
    assert after_removal.cells[Coord(row_idx=3, col_idx=3, entry_idx=30)].value == 0

    assert after_removal.cells[coord_2].value == 8
    assert after_removal.cells[coord_2].allowed_values == []


create_all_coords_test()
get_coords_in_block_test()
get_coords_in_block_of_coord_test()
create_coord_to_all_coords_in_row_col_or_block_test()
create_empty_grid_test()
set_value_in_grid_test()

print("all tests passed")

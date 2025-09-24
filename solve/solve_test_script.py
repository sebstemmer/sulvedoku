from __future__ import annotations

from grid.grid import create_empty_grid, str_to_grid, Coord, Grid, get_entry_idx, set_value_in_grid, \
    all_coords_0_to_80, is_equal
from solve.solve import recursively_solve_trivial_solutions, create_filled, SolutionPathNode, \
    ordered_guess_strategy, random_guess_strategy, smallest_allowed_guess_strategy, solve_grid


def find_trivial_solutions_test() -> None:
    grid: Grid = create_empty_grid()

    for col_idx in range(9):
        value = col_idx + 1
        if col_idx != 4 and col_idx != 7:
            grid = set_value_in_grid(
                grid=grid,
                coord=Coord(
                    row_idx=0,
                    col_idx=col_idx,
                    entry_idx=get_entry_idx(
                        row_idx=0,
                        col_idx=col_idx
                    )),
                value=value,
            )

    grid = set_value_in_grid(
        grid=grid,
        coord=Coord(
            row_idx=8,
            col_idx=7,
            entry_idx=get_entry_idx(
                row_idx=8,
                col_idx=7
            )
        ),
        value=5
    )

    node = SolutionPathNode(
        grid=grid,
        at=None
    )

    handled_trivial_solutions: tuple[SolutionPathNode | None, int] = recursively_solve_trivial_solutions(
        node=node,
        depth=0
    )

    trivial_solution_is_5_coord = Coord(
        row_idx=0,
        col_idx=4,
        entry_idx=get_entry_idx(row_idx=0, col_idx=4)
    )

    trivial_solution_is_8_coord = Coord(
        row_idx=0,
        col_idx=7,
        entry_idx=get_entry_idx(row_idx=0, col_idx=7)
    )

    assert handled_trivial_solutions[0].grid.cells[trivial_solution_is_5_coord].value == 5
    assert handled_trivial_solutions[0].grid.cells[trivial_solution_is_8_coord].value == 8

    assert handled_trivial_solutions[0].at is not None
    assert handled_trivial_solutions[0].at.is_trivial

    assert handled_trivial_solutions[0].at.previous_node.at is not None
    assert handled_trivial_solutions[0].at.previous_node.at.is_trivial


def solve_grid_test() -> None:
    grid_str: str = "068700900004000071031809050305080100046005007007304092602001005003020600059030028"
    solution_str: str = "568712943924653871731849256395287164246195387817364592682971435473528619159436728"

    grid = str_to_grid(
        grid_as_str=grid_str
    )

    solution_random: Grid = solve_grid(
        grid=grid,
        guess_strategy=random_guess_strategy
    )

    assert is_equal(grid1=solution_random, grid2=str_to_grid(grid_as_str=solution_str))

    solution_ordered: Grid = solve_grid(
        grid=grid,
        guess_strategy=ordered_guess_strategy
    )

    assert is_equal(grid1=solution_ordered, grid2=str_to_grid(grid_as_str=solution_str))

    solution_smallest_allowed: Grid = solve_grid(
        grid=grid,
        guess_strategy=smallest_allowed_guess_strategy
    )

    assert is_equal(grid1=solution_smallest_allowed, grid2=str_to_grid(grid_as_str=solution_str))


def create_filled_test() -> None:
    filled_by_random: Grid = create_filled(
        guess_strategy=random_guess_strategy
    )

    filled_by_ordered: Grid = create_filled(
        guess_strategy=ordered_guess_strategy
    )

    filled_by_smallest_allowed: Grid = create_filled(
        guess_strategy=smallest_allowed_guess_strategy
    )

    filled_by_ordered_grid: Grid = create_empty_grid()
    filled_by_random_grid: Grid = create_empty_grid()
    filled_by_smallest_allowed_grid: Grid = create_empty_grid()

    for coord in all_coords_0_to_80:
        filled_by_ordered_grid = set_value_in_grid(
            grid=filled_by_ordered_grid,
            coord=coord,
            value=filled_by_ordered.cells[coord].value
        )
        assert filled_by_ordered_grid is not None

        filled_by_random_grid = set_value_in_grid(
            grid=filled_by_random_grid,
            coord=coord,
            value=filled_by_random.cells[coord].value
        )
        assert filled_by_random_grid is not None

        filled_by_smallest_allowed_grid = set_value_in_grid(
            grid=filled_by_smallest_allowed_grid,
            coord=coord,
            value=filled_by_smallest_allowed.cells[coord].value
        )
        assert filled_by_smallest_allowed_grid is not None


find_trivial_solutions_test()
solve_grid_test()
create_filled_test()

print("all tests passed")

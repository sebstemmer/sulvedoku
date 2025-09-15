from __future__ import annotations

from grid.grid import create_empty_grid, str_to_grid, Coord, Grid, get_entry_idx, set_value_in_grid, \
    all_coords_0_to_80
from solve.solve import recursively_solve_trivial_solutions, create_filled, recursively_find_solution, SolutionPathNode, \
    ordered_guess_strategy, random_guess_strategy, smallest_allowed_guess_strategy


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

    handled_trivial_solutions: SolutionPathNode = recursively_solve_trivial_solutions(
        node=node
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

    assert handled_trivial_solutions.grid.cells[trivial_solution_is_5_coord].value == 5
    assert handled_trivial_solutions.grid.cells[trivial_solution_is_8_coord].value == 8

    assert handled_trivial_solutions.at is not None
    assert handled_trivial_solutions.at.is_trivial

    assert handled_trivial_solutions.at.previous_node.at is not None
    assert handled_trivial_solutions.at.previous_node.at.is_trivial


def recursively_find_solution_test() -> None:
    non_unique_grid_str: str = "600000237070080400203000019320600004004000500000041700000506940007008605500000071"

    grid = str_to_grid(
        grid_as_str=non_unique_grid_str
    )

    node = SolutionPathNode(
        grid=grid,
        at=None
    )

    _: SolutionPathNode = recursively_find_solution(
        node=node,
        guess_strategy=random_guess_strategy,
        max_go_back_depth=None
    )

    pass


def create_filled_test() -> None:
    filled_by_random: SolutionPathNode = create_filled(
        max_go_back_depth=2,
        guess_strategy=random_guess_strategy
    )

    filled_by_ordered: SolutionPathNode = create_filled(
        max_go_back_depth=2,
        guess_strategy=ordered_guess_strategy
    )

    filled_by_smallest_allowed: SolutionPathNode = create_filled(
        max_go_back_depth=2,
        guess_strategy=smallest_allowed_guess_strategy
    )

    filled_by_ordered_grid: Grid = create_empty_grid()
    filled_by_random_grid: Grid = create_empty_grid()
    filled_by_smallest_allowed_grid: Grid = create_empty_grid()

    for coord in all_coords_0_to_80:
        filled_by_ordered_grid = set_value_in_grid(
            grid=filled_by_ordered_grid,
            coord=coord,
            value=filled_by_ordered.grid.cells[coord].value
        )
        assert filled_by_ordered_grid is not None

        filled_by_random_grid = set_value_in_grid(
            grid=filled_by_random_grid,
            coord=coord,
            value=filled_by_random.grid.cells[coord].value
        )
        assert filled_by_random_grid is not None

        filled_by_smallest_allowed_grid = set_value_in_grid(
            grid=filled_by_smallest_allowed_grid,
            coord=coord,
            value=filled_by_smallest_allowed.grid.cells[coord].value
        )
        assert filled_by_smallest_allowed_grid is not None


find_trivial_solutions_test()
recursively_find_solution_test()
create_filled_test()

print("all tests passed")

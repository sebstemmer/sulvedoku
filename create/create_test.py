import sys

from create.create_core import check_if_has_unique_solution, create_filled, create_random_filled
from grid.grid import create_empty_grid, str_to_grid, Coord, Grid, get_entry_idx, remove_value_from_grid, \
    all_coords_0_to_80, set_value_in_grid
from solve.path import SolutionPathNode, ordered_guess_strategy, random_guess_strategy

sys.setrecursionlimit(int(1e4))


def create_random_filled_test() -> None:
    filled_by_random: SolutionPathNode = create_random_filled(
        max_go_back_depth=-1,
        guess_strategy=random_guess_strategy
    )

    filled_by_ordered_grid: Grid = create_empty_grid()
    filled_by_random_grid: Grid = create_empty_grid()
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


def check_if_has_unique_solution_test() -> None:
    filled: SolutionPathNode = create_filled(
        max_go_back_depth=-1,
        guess_strategy=ordered_guess_strategy
    )

    assert check_if_has_unique_solution(
        grid=filled.grid,
        solution_grid=filled.grid
    )

    grid = filled.grid
    for col_idx in range(0, 9):
        coord = Coord(
            row_idx=0,
            col_idx=col_idx,
            entry_idx=get_entry_idx(
                row_idx=0,
                col_idx=col_idx
            )
        )

        grid = remove_value_from_grid(
            grid=grid,
            coord=coord
        )

    assert check_if_has_unique_solution(
        grid=grid,
        solution_grid=filled.grid
    )

    for col_idx in range(0, 9):
        coord = Coord(
            row_idx=1,
            col_idx=col_idx,
            entry_idx=get_entry_idx(
                row_idx=1,
                col_idx=col_idx
            )
        )

        grid = remove_value_from_grid(
            grid=grid,
            coord=coord
        )

    assert not check_if_has_unique_solution(
        grid=grid,
        solution_grid=filled.grid
    )

    non_unique_grid_str: str = "600000237070080400203000019320600004004000500000041700000506940007008605500000071"

    non_unique_grid: Grid = str_to_grid(
        grid_as_str=non_unique_grid_str
    )

    non_unique_solution_grid = str_to_grid(
        grid_as_str="658914237971382456243765819329657184714893562865241793132576948497138625586429371"
    )

    assert not check_if_has_unique_solution(
        grid=non_unique_grid,
        solution_grid=non_unique_solution_grid
    )

    another_non_unique_grid: Grid = str_to_grid(
        grid_as_str="020009004000402800010000326007085001300000000000200700500001007091700000700000040"
    )

    another_non_unique_solution_grid: Grid = str_to_grid(
        grid_as_str="823169574675432819419578326947385261352617498186294753564821937291743685738956142"
    )

    assert not check_if_has_unique_solution(
        grid=another_non_unique_grid,
        solution_grid=another_non_unique_solution_grid
    )


create_filled_test()
check_if_has_unique_solution_test()

print("all tests passed")

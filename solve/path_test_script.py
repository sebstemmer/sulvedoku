from __future__ import annotations

from core.grid import create_empty_grid, str_to_grid, remove_values_from_grid, Coord, Cell, Grid, grid_to_str
from core.path import recursively_solve_trivial_solutions, create_start_node, create_filled, \
    check_if_has_unique_solution, recursively_find_solution, SolutionPathNode, find_next_coord_and_value_for_ordered, \
    find_next_coord_and_value_for_random


def create_valid_filled() -> Grid:
    grid: Grid = create_empty_grid()

    for row_idx in range(0, 9):
        row_square_idx = (row_idx // 3)
        value_start = row_square_idx + (row_idx - 3 * row_square_idx) * 3
        values = [(a % 9) + 1 for a in range(value_start, value_start + 9)]
        for col_idx in range(9):
            value = values[col_idx]
            coord = Coord(row_idx=row_idx, col_idx=col_idx)
            grid.cells[coord] = Cell(
                value=value,
                allowed_values=[]
            )

    return grid


def find_trivial_solutions_test() -> None:
    grid: Grid = create_empty_grid()

    for col_idx in range(9):
        value = col_idx + 1
        if col_idx != 4 and col_idx != 7:
            grid.cells[Coord(row_idx=0, col_idx=col_idx)] = Cell(
                value=value,
                allowed_values=[]
            )

    trivial_solution_is_5_coord = Coord(row_idx=0, col_idx=4)

    grid.cells[trivial_solution_is_5_coord] = Cell(
        value=0,
        allowed_values=[5]
    )

    trivial_solution_is_8_coord = Coord(row_idx=0, col_idx=7)

    grid.cells[trivial_solution_is_8_coord] = Cell(
        value=0,
        allowed_values=[8]
    )

    node = create_start_node(
        grid=grid,
        max_go_back_depth=None
    )

    handled_trivial_solutions: SolutionPathNode = recursively_solve_trivial_solutions(
        node=node,
        recursion_depth=0
    )

    assert handled_trivial_solutions.grid.cells[trivial_solution_is_5_coord].value == 5
    assert handled_trivial_solutions.grid.cells[trivial_solution_is_8_coord].value == 8

    assert handled_trivial_solutions.at is not None
    assert handled_trivial_solutions.at.is_trivial

    assert handled_trivial_solutions.at.previous_node.at is not None
    assert handled_trivial_solutions.at.previous_node.at.is_trivial


def recursively_find_solution_test() -> None:
    # non_unique_grid_str: str = "600000237070080400203000019320600004004000500000041700000506940007008605500000071"
    non_unique_grid_str: str = "068700900004000071031809050305080100046005007007304092602001005003020600059030028"

    grid = str_to_grid(
        grid_as_str=non_unique_grid_str
    )

    node = create_start_node(
        grid=grid,
        max_go_back_depth=None
    )

    sol: SolutionPathNode = recursively_find_solution(
        node=node,
        guess_strategy=find_next_coord_and_value_for_random,
        recursion_depth=0
    )

    print(grid_to_str(sol.grid, "\n", " "))

    pass


def create_filled_test() -> None:
    filled_ordered: SolutionPathNode = create_filled(
        max_go_back_depth=-1,
        guess_strategy=find_next_coord_and_value_for_ordered
    )

    assert filled_ordered.grid.is_valid

    filled_random: SolutionPathNode = create_filled(
        max_go_back_depth=-1,
        guess_strategy=find_next_coord_and_value_for_random
    )

    assert filled_random.grid.is_valid



def check_if_has_unique_solution_test() -> None:
    filled: SolutionPathNode = create_filled(
        max_go_back_depth=-1,
        guess_strategy=find_next_coord_and_value_for_ordered
    )

    assert check_if_has_unique_solution(
        node=filled,
        solution_grid=filled.grid
    )

    removed_first_row_grid = remove_values_from_grid(
        grid=filled.grid,
        coords=[
            Coord(
                row_idx=0,
                col_idx=col_idx
            ) for col_idx in range(9)
        ]
    )

    removed_first_row = create_start_node(
        grid=removed_first_row_grid,
        max_go_back_depth=None
    )

    assert check_if_has_unique_solution(
        node=removed_first_row,
        solution_grid=filled.grid
    )

    removed_first_two_rows_grid = remove_values_from_grid(
        grid=filled.grid,
        coords=[
                   Coord(
                       row_idx=0,
                       col_idx=col_idx
                   ) for col_idx in range(9)
               ] + [
                   Coord(
                       row_idx=1,
                       col_idx=col_idx
                   ) for col_idx in range(9)
               ]
    )

    removed_first_two_rows = create_start_node(
        grid=removed_first_two_rows_grid,
        max_go_back_depth=None
    )

    assert not check_if_has_unique_solution(
        node=removed_first_two_rows,
        solution_grid=filled.grid
    )

    non_unique_grid_str: str = "600000237070080400203000019320600004004000500000041700000506940007008605500000071"

    non_unique_grid: Grid = str_to_grid(
        grid_as_str=non_unique_grid_str
    )

    non_unique: SolutionPathNode = create_start_node(
        grid=non_unique_grid,
        max_go_back_depth=None
    )

    non_unique_solution_grid = str_to_grid(
        grid_as_str="658914237971382456243765819329657184714893562865241793132576948497138625586429371"
    )

    assert not check_if_has_unique_solution(
        node=non_unique,
        solution_grid=non_unique_solution_grid
    )

    another_non_unique_grid: Grid = str_to_grid(
        grid_as_str="020009004000402800010000326007085001300000000000200700500001007091700000700000040"
    )

    another_non_unique: SolutionPathNode = create_start_node(
        grid=another_non_unique_grid,
        max_go_back_depth=None
    )

    another_non_unique_solution_grid: Grid = str_to_grid(
        grid_as_str="823169574675432819419578326947385261352617498186294753564821937291743685738956142"
    )

    assert not check_if_has_unique_solution(
        node=another_non_unique,
        solution_grid=another_non_unique_solution_grid
    )


#find_trivial_solutions_test()
#create_filled_test()
recursively_find_solution_test()
#check_if_has_unique_solution_test()

print("all tests passed")

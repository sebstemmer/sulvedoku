from __future__ import annotations
import utils as utils
from typing import Optional
import utils

from grid import create_empty_grid, create_coord_to_all_coords_in_row_col_or_square, set_value_in_grid, is_valid, str_to_grid, Coord, Cell
from path import create_fill_path, find_trivial_solutions, create_start_node, find_next_fill_path_idx, create_filled, check_if_has_unique_solution, recursively_find_solution, FillPathCreationMethod, SolutionPathNode, At

all_coords: list[Coord] = list(
    create_coord_to_all_coords_in_row_col_or_square().keys()
)
coord_to_all_coords_in_row_col_or_square: dict[
    Coord, set[Coord]
] = create_coord_to_all_coords_in_row_col_or_square()


def create_valid_filled() -> dict[Coord, Cell]:
    grid: dict[Coord, Cell] = create_empty_grid(all_coords=all_coords)

    for row_idx in range(0, 9):
        row_square_idx = (row_idx // 3)
        value_start = row_square_idx + (row_idx - 3 * row_square_idx) * 3
        values = [(a % 9)+1 for a in range(value_start, value_start+9)]
        for col_idx in range(9):
            value = values[col_idx]
            coord = Coord(row_idx=row_idx, col_idx=col_idx)
            grid[coord] = Cell(
                value=value,
                allowed_values=[]
            )

    return grid


def create_fill_path_test() -> None:
    empty_grid = create_empty_grid(
        all_coords=all_coords
    )

    random_fill_path: list[Coord] = create_fill_path(
        grid=empty_grid,
        method=FillPathCreationMethod.ORDERED
    )

    assert len(set(random_fill_path)) == 81
    assert random_fill_path[8] == Coord(row_idx=0, col_idx=8)

    random_fill_path: list[Coord] = create_fill_path(
        grid=empty_grid,
        method=FillPathCreationMethod.RANDOM
    )

    assert len(set(random_fill_path)) == 81
    assert Coord(row_idx=0, col_idx=8) in set(random_fill_path)


def find_trivial_solutions_test() -> None:

    grid = create_empty_grid(
        all_coords=all_coords
    )

    for col_idx in range(9):
        value = col_idx + 1
        if col_idx != 4 and col_idx != 7:
            grid[Coord(row_idx=0, col_idx=col_idx)] = Cell(
                value=value,
                allowed_values=[]
            )

    trivial_solution_is_5_coord = Coord(row_idx=0, col_idx=4)

    grid[trivial_solution_is_5_coord] = Cell(
        value=0,
        allowed_values=[5]
    )

    trivial_solution_is_8_coord = Coord(row_idx=0, col_idx=7)

    grid[trivial_solution_is_8_coord] = Cell(
        value=0,
        allowed_values=[8]
    )

    node = create_start_node(
        grid=grid,
        fill_path=[],
        max_go_back_depth=None
    )

    handled_trivial_solutions: SolutionPathNode = find_trivial_solutions(
        node=node,
        depth=0,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )

    assert handled_trivial_solutions.grid[trivial_solution_is_5_coord].value == 5
    assert handled_trivial_solutions.grid[trivial_solution_is_8_coord].value == 8

    assert handled_trivial_solutions.at is not None
    assert handled_trivial_solutions.at.is_trivial

    assert handled_trivial_solutions.at.previous_node.at is not None
    assert handled_trivial_solutions.at.previous_node.at.is_trivial


def find_next_fill_path_idx_test() -> None:
    grid: dict[Coord, Cell] = create_empty_grid(
        all_coords=all_coords
    )

    fill_path = create_fill_path(
        grid=grid,
        method=FillPathCreationMethod.ORDERED
    )

    node = create_start_node(
        grid=grid,
        fill_path=fill_path,
        max_go_back_depth=None
    )

    next_fill_path_idx: Optional[int] = find_next_fill_path_idx(
        node=node
    )

    assert next_fill_path_idx == 0

    next_grid = set_value_in_grid(
        grid=grid,
        coord=fill_path[0],
        value=5,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )

    next_next_grid = set_value_in_grid(
        grid=next_grid,
        coord=fill_path[1],
        value=3,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )

    next: SolutionPathNode = SolutionPathNode(
        grid=next_next_grid,
        fill_path=fill_path,
        at=At(
            coord=fill_path[0],
            value_tries=[5],
            fill_path_idx=0,
            previous_node=node,
            is_trivial=False
        ),
        depth=0,
        max_go_back_depth=None
    )

    next_fill_path_idx: Optional[int] = find_next_fill_path_idx(
        node=next
    )

    assert next_fill_path_idx == 2

    value_at_first_fill_path_idx_grid: dict[Coord, Cell] = create_empty_grid(
        all_coords=all_coords
    )

    fill_path: list[Coord] = create_fill_path(
        grid=value_at_first_fill_path_idx_grid,
        method=FillPathCreationMethod.ORDERED
    )

    value_at_first_fill_path_idx_grid: dict[Coord, Cell] = set_value_in_grid(
        grid=value_at_first_fill_path_idx_grid,
        coord=fill_path[0],
        value=5,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )

    value_at_first_fill_path_idx: SolutionPathNode = create_start_node(
        grid=value_at_first_fill_path_idx_grid,
        fill_path=fill_path,
        max_go_back_depth=None
    )

    next_fill_path_idx: Optional[int] = find_next_fill_path_idx(
        node=value_at_first_fill_path_idx
    )

    assert next_fill_path_idx == 1


def recursively_find_solution_test() -> None:
    non_unique_grid_str: str = "600000237070080400203000019320600004004000500000041700000506940007008605500000071"

    grid = str_to_grid(
        grid_as_str=non_unique_grid_str,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )

    fill_path = create_fill_path(
        grid=grid,
        method=FillPathCreationMethod.RANDOM
    )

    node = create_start_node(
        grid=grid,
        fill_path=fill_path,
        max_go_back_depth=None
    )

    _: SolutionPathNode = recursively_find_solution(
        node=node,
        depth=0,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )

    pass


def create_filled_test() -> None:
    filled_ordered: SolutionPathNode = create_filled(
        method=FillPathCreationMethod.ORDERED,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square,
        max_go_back_depth=-1
    )

    assert is_valid(
        grid=filled_ordered.grid,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )

    filled_random: SolutionPathNode = create_filled(
        method=FillPathCreationMethod.RANDOM,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square,
        max_go_back_depth=-1
    )

    assert is_valid(
        grid=filled_random.grid,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )


def check_if_has_unique_solution_test() -> None:
    # filled: SolutionPathNode = create_filled(
    #     method=FillPathCreationMethod.ORDERED,
    #     coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square,
    #     max_go_back_depth=-1
    # )

    # assert check_if_has_unique_solution(
    #     node=filled,
    #     solution=filled.grid,
    #     coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    # )

    # removed_first_row_grid = remove_values_from_grid(
    #     grid=filled.grid,
    #     coords=[
    #         Coord(
    #             row_idx=0,
    #             col_idx=col_idx
    #         ) for col_idx in range(9)
    #     ],
    #     coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    # )

    # removed_first_row_fill_path = create_fill_path(
    #     removed_first_row_grid,
    #     method=FillPathCreationMethod.RANDOM
    # )

    # removed_first_row = create_start_node(
    #     grid=removed_first_row_grid,
    #     fill_path=removed_first_row_fill_path,
    #     max_go_back_depth=None
    # )

    # assert check_if_has_unique_solution(
    #     node=removed_first_row,
    #     solution=filled.grid,
    #     coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    # )

    # removed_first_two_rows_grid = remove_values_from_grid(
    #     grid=filled.grid,
    #     coords=[
    #         Coord(
    #             row_idx=0,
    #             col_idx=col_idx
    #         ) for col_idx in range(9)
    #     ] + [
    #         Coord(
    #             row_idx=1,
    #             col_idx=col_idx
    #         ) for col_idx in range(9)
    #     ],
    #     coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    # )

    # removed_first_two_rows_fill_path = create_fill_path(
    #     removed_first_two_rows_grid,
    #     method=FillPathCreationMethod.RANDOM
    # )

    # removed_first_two_rows = create_start_node(
    #     grid=removed_first_two_rows_grid,
    #     fill_path=removed_first_two_rows_fill_path,
    #     max_go_back_depth=None
    # )

    # assert not check_if_has_unique_solution(
    #     node=removed_first_two_rows,
    #     solution=filled.grid,
    #     coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    # )

    non_unique_grid_str: str = "600000237070080400203000019320600004004000500000041700000506940007008605500000071"

    non_unique_grid: dict[Coord, Cell] = str_to_grid(
        grid_as_str=non_unique_grid_str,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )

    non_unique_fill_path: list[Coord] = create_fill_path(
        grid=non_unique_grid,
        method=FillPathCreationMethod.RANDOM
    )

    non_unique: SolutionPathNode = create_start_node(
        grid=non_unique_grid,
        fill_path=non_unique_fill_path,
        max_go_back_depth=None
    )

    non_unique_solution_grid = str_to_grid(
        grid_as_str="658914237971382456243765819329657184714893562865241793132576948497138625586429371",
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )

    assert not check_if_has_unique_solution(
        node=non_unique,
        solution=non_unique_solution_grid,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )


create_fill_path_test()
find_trivial_solutions_test()
find_next_fill_path_idx_test()
create_filled_test()
recursively_find_solution_test()
check_if_has_unique_solution_test()

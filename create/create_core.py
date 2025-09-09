from __future__ import annotations

import random
from typing import NamedTuple, Optional

from core.grid import Coord, set_value_in_grid, Grid, \
    remove_value_from_grid, all_coords_0_to_80
from core.path import SolutionPathNode, recursively_solve_trivial_solutions, create_node, \
    find_next_coord_and_value_for_smallest_allowed, solve_grid, GoBackFailed, create_filled, \
    find_next_coord_and_value_for_random


class At(NamedTuple):
    coord: Coord
    tries: set[Coord]
    previous_node: RemovePathNode


class RemovePathNode(NamedTuple):
    grid: Grid
    filled_coords: tuple[Coord, ...]
    at: Optional[At]
    selection_depth: int


class MaxSelectionDepthReached(Exception):
    pass


def create_start_node(
        filled_grid: Grid,
) -> RemovePathNode:
    return RemovePathNode(
        grid=filled_grid,
        filled_coords=all_coords_0_to_80,
        at=None,
        selection_depth=0
    )


def check_if_has_unique_solution(
        grid: Grid,
        solution_grid: Grid
) -> bool:
    if len(grid.empty_coords) == 0:
        return True

    node = create_node(
        max_go_back_depth=None,
        grid=grid,
        at=None
    )

    after_trivial: SolutionPathNode | None = recursively_solve_trivial_solutions(node)

    if after_trivial is None:
        raise ValueError("grid is not valid")

    next_coord_and_value: tuple[Coord, int] | None = find_next_coord_and_value_for_smallest_allowed(
        grid=after_trivial.grid)

    if next_coord_and_value is None:
        return True

    next_coord: Coord = next_coord_and_value[0]

    allowed_values: tuple[int, ...] = after_trivial.grid.cells[next_coord].allowed_values

    if len(allowed_values) == 1:
        raise RuntimeError("trivial solutions have been solved before")

    solution_value: int = solution_grid.cells[next_coord].value

    potential_other_solution_values = [a for a in allowed_values if a != solution_value]

    for value in potential_other_solution_values:
        potential_other_solution_grid: Grid | None = set_value_in_grid(
            grid=after_trivial.grid,
            coord=next_coord,
            value=value
        )

        if potential_other_solution_grid is not None:
            try:
                _ = solve_grid(
                    grid=potential_other_solution_grid,
                    max_go_back_depth=None,
                    guess_strategy=find_next_coord_and_value_for_smallest_allowed
                )
                return False
            except GoBackFailed:
                pass

    new_grid = set_value_in_grid(
        grid=after_trivial.grid,
        coord=next_coord,
        value=solution_value
    )

    return check_if_has_unique_solution(
        grid=new_grid,
        solution_grid=solution_grid
    )


def go_back_to_previous_node_and_try_other_coord(
        selection_depth: int,
        node: RemovePathNode,
        solution_grid: Grid,
        num_filled_target: int,
        max_selection_depth: int
):
    at: At | None = node.at

    if node.at is None:
        raise ValueError("solution grid not valid")

    return try_other_coord(
        grid=at.previous_node.grid,
        already_tried=at.tries,
        filled_coords=node.at.previous_node.filled_coords,
        selection_depth=selection_depth,
        node=node.at.previous_node,
        solution_grid=solution_grid,
        num_filled_target=num_filled_target,
        max_selection_depth=max_selection_depth
    )


def try_other_coord(
        grid: Grid,
        already_tried: set[Coord],
        filled_coords: tuple[Coord, ...],
        selection_depth: int,
        node: RemovePathNode,
        solution_grid: Grid,
        num_filled_target: int,
        max_selection_depth: int
) -> RemovePathNode:
    pool: list[Coord] = [c for c in filled_coords if c not in already_tried]

    if len(pool) == 0:
        return go_back_to_previous_node_and_try_other_coord(
            selection_depth=selection_depth,
            node=node,
            solution_grid=solution_grid,
            num_filled_target=num_filled_target,
            max_selection_depth=max_selection_depth
        )

    coord: Coord = random.choice(pool)

    new_selection_depth = selection_depth + 1

    if new_selection_depth > max_selection_depth:
        raise MaxSelectionDepthReached

    new_grid: Grid = remove_value_from_grid(
        grid=grid,
        coord=coord
    )

    has_unique_solution: bool = check_if_has_unique_solution(
        grid=new_grid,
        solution_grid=solution_grid
    )

    new_tries = already_tried | {coord}

    if has_unique_solution:
        new_node = RemovePathNode(
            grid=new_grid,
            filled_coords=tuple([c for c in node.filled_coords if c != coord]),
            at=At(
                coord=coord,
                tries=new_tries,
                previous_node=node
            ),
            selection_depth=new_selection_depth
        )
        return recursively_remove_values(
            node=new_node,
            solution_grid=solution_grid,
            num_filled_target=num_filled_target,
            max_selection_depth=max_selection_depth
        )
    else:
        return try_other_coord(
            grid=grid,
            already_tried=new_tries,
            filled_coords=filled_coords,
            selection_depth=new_selection_depth,
            node=node,
            solution_grid=solution_grid,
            num_filled_target=num_filled_target,
            max_selection_depth=max_selection_depth
        )


def recursively_remove_values(
        node: RemovePathNode,
        solution_grid: Grid,
        num_filled_target: int,
        max_selection_depth: int
) -> RemovePathNode:
    if len(node.filled_coords) <= num_filled_target:
        return node

    return try_other_coord(
        grid=node.grid,
        already_tried=set(),
        filled_coords=node.filled_coords,
        selection_depth=node.selection_depth,
        node=node,
        solution_grid=solution_grid,
        num_filled_target=num_filled_target,
        max_selection_depth=max_selection_depth
    )


def create_grid(
        num_filled_target: int,
        max_selection_depth: int,
) -> RemovePathNode:
    filled_grid: Grid = create_filled(
        max_go_back_depth=-1,
        guess_strategy=find_next_coord_and_value_for_random
    ).grid

    try:
        return recursively_remove_values(
            node=create_start_node(filled_grid),
            solution_grid=filled_grid,
            num_filled_target=num_filled_target,
            max_selection_depth=max_selection_depth
        )
    except MaxSelectionDepthReached:
        print("restart")
        return create_grid(
            num_filled_target=num_filled_target,
            max_selection_depth=max_selection_depth,
        )

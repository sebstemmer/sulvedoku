from __future__ import annotations

import random
import sys
from typing import NamedTuple, Optional

from grid.grid import Coord, set_value_in_grid, Grid, remove_value_from_grid
from solve.solve import SolutionPathNode, recursively_solve_trivial_solutions, smallest_allowed_guess_strategy, \
    solve_grid, GoBackFailed, create_filled, random_guess_strategy


class At(NamedTuple):
    """
        Describes which node of the Sudoku remove path is handled.

        Attributes:
            coord (Coord):
                Coordinate of grid at which a value was removed.
            tries (frozenset[int]):
                Already tried these coordinates for removal on other attempts.
            previous_node (SolutionPathNode):
                Previous node of remove path.
    """
    coord: Coord
    tries: set[Coord]
    previous_node: RemovePathNode


class RemovePathNode(NamedTuple):
    """
        Creating a Sudoku is modeled as a linked list (state graph). This list is called remove path.
        When a coordinate is tried a new grid and hence a new node on the remove path is created.

        Attributes:
            grid (Grid):
                To each node on the remove path belongs a grid.
            at (Optional[At]):
                None at the start, afterwards describes the last removed coordinate.

    """
    grid: Grid
    at: Optional[At]


class MaxRemoveDepthReached(Exception):
    """
        Every time a coordinate value is removed the depth increases by one. Raised when the maximum depth is reached.
    """
    pass


def check_if_has_unique_solution(
        grid: Grid,
        solution_grid: Grid
) -> bool:
    """
        Test if the provided Sudoku grid has a unique solution.

        Args:
            grid (Grid):
                Grid under test.
            solution_grid (Grid):
                One known valid solution.

        Returns:
            bool:
                True if the grid has a unique solution, False otherwise.
    """
    if len(grid.empty_coords) == 0:
        return True

    node = SolutionPathNode(
        grid=grid,
        at=None
    )

    after_trivial, _ = recursively_solve_trivial_solutions(node=node, depth=0)

    if after_trivial is None:
        raise ValueError("grid is not valid")

    if len(after_trivial.grid.empty_coords) == 0:
        return True

    next_coord_and_value: tuple[Coord, int] = smallest_allowed_guess_strategy(
        grid=after_trivial.grid
    )

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
                    guess_strategy=smallest_allowed_guess_strategy
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
        remove_depth: int,
        node: RemovePathNode,
        solution_grid: Grid,
        num_filled_target: int,
        max_remove_depth: int
) -> RemovePathNode:
    """
        Go back to previous node and try another coordinate, because removing the tried coordinate has lead to a
        non-unique solution.

        Args:
            remove_depth (int):
                Current remove depth.
            node (RemovePathNode):
                Node to go back from.
            solution_grid (Grid):
                One known valid solution.
            num_filled_target (int):
                Number of filled cells in the target Sudoku grid that is created.
            max_remove_depth (int):
                The maximum remove depth, if reached the Sudoku creating is restarted.

        Returns:
            RemovePathNode:
                Node with final created Sudoku grid.
    """
    at: At | None = node.at

    if node.at is None:
        raise ValueError("solution grid not valid")

    return try_other_coord(
        grid=at.previous_node.grid,
        already_tried=at.tries,
        remove_depth=remove_depth,
        node=node.at.previous_node,
        solution_grid=solution_grid,
        num_filled_target=num_filled_target,
        max_remove_depth=max_remove_depth
    )


def try_other_coord(
        grid: Grid,
        already_tried: set[Coord],
        remove_depth: int,
        node: RemovePathNode,
        solution_grid: Grid,
        num_filled_target: int,
        max_remove_depth: int
) -> RemovePathNode:
    """
        Try another coordinate.

        Args:
            grid (Grid):
                Current grid.
            already_tried (set[Coord]):
                Already tried these coordinates in previous attempts.
            remove_depth (int):
                Current remove depth.
            node (RemovePathNode):
                Current node.
            solution_grid (Grid):
                One known valid solution.
            num_filled_target (int):
                Number of filled cells in the target Sudoku grid that is created.
            max_remove_depth (int):
                The maximum remove depth, if reached the Sudoku creating is restarted.

        Returns:
            RemovePathNode:
                Node with final created Sudoku grid.
    """
    pool: list[Coord] = [c for c in grid.filled_coords if c not in already_tried]

    if len(pool) == 0:
        return go_back_to_previous_node_and_try_other_coord(
            remove_depth=remove_depth,
            node=node,
            solution_grid=solution_grid,
            num_filled_target=num_filled_target,
            max_remove_depth=max_remove_depth
        )

    coord: Coord = random.choice(pool)

    new_remove_depth = remove_depth + 1

    if new_remove_depth > max_remove_depth:
        raise MaxRemoveDepthReached

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
            at=At(
                coord=coord,
                tries=new_tries,
                previous_node=node
            )
        )
        return recursively_remove_values(
            node=new_node,
            remove_depth=new_remove_depth,
            solution_grid=solution_grid,
            num_filled_target=num_filled_target,
            max_remove_depth=max_remove_depth
        )
    else:
        return try_other_coord(
            grid=grid,
            already_tried=new_tries,
            remove_depth=new_remove_depth,
            node=node,
            solution_grid=solution_grid,
            num_filled_target=num_filled_target,
            max_remove_depth=max_remove_depth
        )


def recursively_remove_values(
        node: RemovePathNode,
        remove_depth: int,
        solution_grid: Grid,
        num_filled_target: int,
        max_remove_depth: int
) -> RemovePathNode:
    """
        Recursive function that removes values from a Sudoku to create a Sudoku with a specific number of filled cells.

        Args:
            node (RemovePathNode):
                Current node.
            remove_depth (int):
                Current remove depth.
            solution_grid (Grid):
                One known valid solution.
            num_filled_target (int):
                Number of filled cells in the target Sudoku grid that is created.
            max_remove_depth (int):
                The maximum remove depth, if reached the Sudoku creating is restarted.

        Returns:
            RemovePathNode:
                Node with final created Sudoku grid.
    """
    if len(node.grid.filled_coords) <= num_filled_target:
        return node

    return try_other_coord(
        grid=node.grid,
        already_tried=set(),
        remove_depth=remove_depth,
        node=node,
        solution_grid=solution_grid,
        num_filled_target=num_filled_target,
        max_remove_depth=max_remove_depth
    )


def create_grid(
        num_filled_target: int,
        max_remove_depth: int = 130,
) -> Grid:
    """
        Create a Sudoku with a specific number of filled cells.

        Args:
            num_filled_target (int):
                Number of filled cells in the target Sudoku grid that is created.
            max_remove_depth (int):
                The maximum remove depth, if reached the Sudoku creating is restarted.

        Returns:
            Grid:
                Created Sudoku grid.
    """
    if 81 - num_filled_target > max_remove_depth:
        raise ValueError("max_depth is too small")

    sys.setrecursionlimit(int(1e7))

    filled_grid: Grid = create_filled(
        max_depth=150,
        guess_strategy=random_guess_strategy
    )

    try:
        return recursively_remove_values(
            node=RemovePathNode(
                grid=filled_grid,
                at=None
            ),
            remove_depth=0,
            solution_grid=filled_grid,
            num_filled_target=num_filled_target,
            max_remove_depth=max_remove_depth
        ).grid
    except MaxRemoveDepthReached:
        return create_grid(
            num_filled_target=num_filled_target,
            max_remove_depth=max_remove_depth,
        )

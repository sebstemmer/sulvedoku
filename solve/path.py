from __future__ import annotations

import random
from enum import Enum
from typing import NamedTuple, Optional, Callable

from core.grid import Coord, Cell, create_empty_grid, set_value_in_grid, \
    Grid


class At(NamedTuple):
    coord: Coord
    tries: tuple[int, ...]
    previous_node: SolutionPathNode
    is_trivial: bool


class SolutionPathNode(NamedTuple):
    """
        max_go_back_depth: Maximum depth of going back when backtracking
         (None means no maximum depth, -1 means  no backtracking).
    """
    grid: Grid
    at: Optional[At]
    max_go_back_depth: Optional[int]  # todo in function


class FillPathCreationMethod(Enum):
    ORDERED = 1
    RANDOM = 2
    SMALLEST_ALLOWED = 3


class MaxGoBackDepthReached(Exception):
    pass


# todo: rename no solution found
class GoBackFailed(Exception):
    pass


class SolutionNotUnique(Exception):
    def __init__(self, depth: int):
        self.depth: int = depth
        super().__init__()


class MaxRemoveCluesDepthReached(Exception):
    def __init__(
            self,
            grid_to_remove: list[tuple[dict[Coord, Cell], Coord]]
    ):
        self.grid_to_remove: list[
            tuple[
                dict[Coord, Cell],
                Coord
            ]] = grid_to_remove
        super().__init__()


# todo -> brauch ich das?
def create_node(
        max_go_back_depth: Optional[int],
        grid: Grid,
        at: Optional[At]
) -> SolutionPathNode:
    return SolutionPathNode(
        grid=grid,
        at=at,
        max_go_back_depth=max_go_back_depth
    )


# todo -> brauch ich das?
def create_start_node(
        grid: Grid,
        max_go_back_depth: Optional[int],
) -> SolutionPathNode:
    return SolutionPathNode(
        grid=grid,
        at=None,
        max_go_back_depth=max_go_back_depth
    )


def solve_trivial_solutions(
        node: SolutionPathNode
) -> tuple[SolutionPathNode, bool] | None:
    found_trivial_solutions: bool = False

    previous_node = node
    for coord in node.grid.empty_coords:
        cell = node.grid.cells[coord]
        allowed_values = cell.allowed_values

        if len(allowed_values) == 1:
            found_trivial_solutions = True
            new_grid: Grid | None = set_value_in_grid(
                grid=previous_node.grid,
                coord=coord,
                value=allowed_values[0]
            )

            if new_grid is None:
                return None

            new_node: SolutionPathNode = create_node(
                max_go_back_depth=node.max_go_back_depth,
                grid=new_grid,
                at=At(
                    coord=coord,
                    tries=(allowed_values[0],),
                    previous_node=previous_node,
                    is_trivial=True
                )
            )
            previous_node = new_node

    return previous_node, found_trivial_solutions


def recursively_solve_trivial_solutions(
        node: SolutionPathNode
) -> SolutionPathNode | None:
    solved_trivial_solutions: tuple[SolutionPathNode, bool] | None = solve_trivial_solutions(
        node=node
    )

    if solved_trivial_solutions is None:
        return None

    after_find_trivial_solutions, trivial_solutions_solved = solved_trivial_solutions

    if not trivial_solutions_solved:
        return after_find_trivial_solutions

    return recursively_solve_trivial_solutions(
        node=after_find_trivial_solutions
    )


def get_complete_solution_path(node: SolutionPathNode) -> list[Coord]:
    if node.at is None:
        return []

    return [node.at.coord] + get_complete_solution_path(node.at.previous_node)


def try_other_value(
        coord: Coord,
        grid: Grid,
        already_tried: tuple[int, ...],  # set
        node: SolutionPathNode,
        guess_strategy: Callable[[Grid], tuple[Coord, int]]
) -> SolutionPathNode:
    allowed_values: tuple[int, ...] = grid.cells[coord].allowed_values

    possible_values: list[int] = [a for a in allowed_values if a not in already_tried]

    if len(possible_values) == 0:
        return go_back_to_previous_node_and_try_other_value(
            node=node,
            go_back_depth=0,
            guess_strategy=guess_strategy
        )

    next_try: int = random.choice(possible_values)

    next_try_grid: Grid | None = set_value_in_grid(
        grid=grid,
        coord=coord,
        value=next_try
    )

    new_already_tried: tuple[int, ...] = already_tried + (next_try,)

    if next_try_grid is None:
        return try_other_value(
            coord=coord,
            grid=grid,
            already_tried=new_already_tried,
            node=node,
            guess_strategy=guess_strategy
        )

    next_try_node: SolutionPathNode = create_node(
        max_go_back_depth=node.max_go_back_depth,
        grid=next_try_grid,
        at=At(
            coord=coord,
            tries=new_already_tried,
            previous_node=node,
            is_trivial=False
        )
    )

    return recursively_find_solution(
        node=next_try_node,
        guess_strategy=guess_strategy
    )


def go_back_to_previous_node_and_try_other_value(
        node: SolutionPathNode,
        go_back_depth: int,
        guess_strategy: Callable[[Grid], tuple[Coord, int]]
) -> SolutionPathNode:
    if node.max_go_back_depth is not None and go_back_depth > node.max_go_back_depth:
        raise MaxGoBackDepthReached()

    at: At | None = node.at

    if node.at is None:
        raise GoBackFailed()

    if at.is_trivial:
        return go_back_to_previous_node_and_try_other_value(
            node=at.previous_node,
            go_back_depth=go_back_depth + 1,
            guess_strategy=guess_strategy
        )

    return try_other_value(
        coord=at.coord,
        grid=at.previous_node.grid,
        already_tried=at.tries,
        node=node.at.previous_node,
        guess_strategy=guess_strategy
    )


def remove_already_tried_from_allowed_values(
        allowed_values: list[int],
        already_tried: list[int]
) -> list[int]:
    return [v for v in allowed_values if v not in already_tried]


def get_total_allowed_values(
        grid: Grid,
        coord: Coord,
        already_tried: list[tuple[Coord, int]]
) -> list[int]:
    already_tried_in_coord: set[int] = {a[1] for a in already_tried if a[0] == coord}

    return [a for a in grid.cells[coord].allowed_values if a not in already_tried_in_coord]


def find_next_coord_and_value_for_random(
        grid: Grid
) -> tuple[Coord, int]:
    coord = random.choice(grid.empty_coords)
    allowed_value = random.choice(grid.cells[coord].allowed_values)

    return coord, allowed_value


def find_next_coord_and_value_for_ordered(
        grid: Grid
) -> tuple[Coord, int]:
    sorted_empty_coords = sorted(grid.empty_coords, key=lambda coord: coord.entry_idx)

    coord = sorted_empty_coords[0]
    allowed_value = random.choice(grid.cells[coord].allowed_values)

    return coord, allowed_value


def find_next_coord_and_value_for_smallest_allowed(
        grid: Grid
) -> tuple[Coord, int]:
    min_num_allowed_values: int = 10
    found_coord: Coord | None = None
    found_allowed_values: tuple[int, ...] = ()

    for coord in grid.empty_coords:
        allowed_values: tuple[int, ...] = grid.cells[coord].allowed_values

        len_allowed_values: int = len(allowed_values)

        if len_allowed_values == 1:
            return coord, allowed_values[0]

        if min_num_allowed_values > len_allowed_values:
            min_num_allowed_values = len_allowed_values
            found_coord = coord
            found_allowed_values = allowed_values

    return found_coord, found_allowed_values[0]


def recursively_find_solution(
        node: SolutionPathNode,
        guess_strategy: Callable[
            [Grid], tuple[Coord, int]
        ]
) -> SolutionPathNode:
    """
    Recursive function to solve a (not necessary valid) grid.

    Args:
        node (SolutionPathNode): Node that contains the grid and the other parameters to solve.
        guess_strategy (Callable[Grid, Optional[tuple[Coord, int]]]): Method for finding next coord and value to handle.

    Returns:
        SolutionPathNode: Last node of solution path containing the solution grid.

    Raises:
        MaxGoBackDepthReached: If maximum backtracking depth is reached
        GoBackFailed: If no solution can be found.
    """

    handled_trivial_solutions: SolutionPathNode | None = recursively_solve_trivial_solutions(
        node=node
    )

    if handled_trivial_solutions is None:
        return go_back_to_previous_node_and_try_other_value(
            node=node,
            go_back_depth=0,
            guess_strategy=guess_strategy,
        )

    if len(handled_trivial_solutions.grid.empty_coords) == 0:
        return handled_trivial_solutions

    next_coord_and_value: tuple[Coord, int] = guess_strategy(
        handled_trivial_solutions.grid
    )

    next_grid: Grid | None = set_value_in_grid(
        grid=handled_trivial_solutions.grid,
        coord=next_coord_and_value[0],
        value=next_coord_and_value[1]
    )

    already_tried: tuple[int, ...] = (next_coord_and_value[1],)

    if next_grid is None:
        return try_other_value(
            coord=next_coord_and_value[0],
            grid=handled_trivial_solutions.grid,
            already_tried=already_tried,
            node=handled_trivial_solutions,
            guess_strategy=guess_strategy,
        )

    next_node: SolutionPathNode = create_node(
        max_go_back_depth=handled_trivial_solutions.max_go_back_depth,
        grid=next_grid,
        at=At(
            coord=next_coord_and_value[0],
            tries=already_tried,
            previous_node=handled_trivial_solutions,
            is_trivial=False
        )
    )

    return recursively_find_solution(
        node=next_node,
        guess_strategy=guess_strategy
    )


def solve_grid(
        grid: Grid,
        max_go_back_depth: Optional[int],
        guess_strategy: Callable[
            [Grid], tuple[Coord, int]
        ],
) -> SolutionPathNode:
    """
    Solve a (not necessary valid) grid.

    Args:
        grid (Grid): Grid to solve.
        max_go_back_depth (Optional[int]): Maximum depth of going back when backtracking (None means no maximum depth).
        guess_strategy (Callable[[Grid, list[tuple[Coord, int]]], tuple[Coord, int] | None]): Method for finding next coord and value to handle.

    Returns:
        SolutionPathNode: Last node of solution path containing the solution grid.

    Raises:
        MaxGoBackDepthReached: If maximum backtracking depth is reached
        GoBackFailed: If no solution can be found.
    """
    start: SolutionPathNode = create_start_node(
        grid=grid,
        max_go_back_depth=max_go_back_depth
    )

    return recursively_find_solution(
        node=start,
        guess_strategy=guess_strategy
    )


# todo: useless
def solve_valid_grid(
        grid: Grid,
        guess_strategy: Callable[[Grid], tuple[Coord, int]]
) -> SolutionPathNode:
    """
    Solve a VALID grid.

    Args:
        grid (dict[Coord, Cell]): Grid to solve.
        guess_strategy (Callable[[Grid, list[tuple[Coord, int]]], Optional[tuple[Coord, int]]]): Method for finding next coordinate and value to handle.

    Returns:
        SolutionPathNode: Last node of solution path containing the solution grid.
    """
    return solve_grid(
        grid=grid,
        max_go_back_depth=None,
        guess_strategy=guess_strategy
    )


def solve_valid_grid_until_no_trivial_solutions(
        grid: Grid
) -> SolutionPathNode:
    """
    Solve a VALID grid until there are no trivial solutions left.

    Args:
        grid (dict[Coord, Cell]): Grid to solve.

    Returns:
        SolutionPathNode: Node when all trivial solutions are found.
    """

    node: SolutionPathNode = create_start_node(
        grid=grid,
        max_go_back_depth=None
    )

    solved_trivial_solutions: SolutionPathNode | None = recursively_solve_trivial_solutions(
        node=node
    )

    if solved_trivial_solutions is None:
        raise ValueError("grid is not valid")

    return solved_trivial_solutions


def create_filled(
        max_go_back_depth: Optional[int],
        guess_strategy: Callable[[Grid], tuple[Coord, int]]
) -> SolutionPathNode:
    """
    Create a filled grid.

    Args:
        max_go_back_depth (Optional[int]): Maximum depth of going back when backtracking (None means no maximum depth).
        guess_strategy (Callable[[Grid, list[tuple[Coord, int]]], Optional[tuple[Coord, int]]]): Method for finding next coord and value to handle.

    Returns:
        SolutionPathNode: Last node of solution path containing the filled grid.
    """
    empty_grid: Grid = create_empty_grid()

    try:
        final: SolutionPathNode = solve_grid(
            grid=empty_grid,
            max_go_back_depth=max_go_back_depth,
            guess_strategy=guess_strategy
        )
    except (GoBackFailed, MaxGoBackDepthReached):
        final: SolutionPathNode = create_filled(
            max_go_back_depth=max_go_back_depth,
            guess_strategy=guess_strategy
        )

    return final


def get_path(node: SolutionPathNode, depth: int) -> list[SolutionPathNode]:
    init_path: list[SolutionPathNode] = [node] if depth == 0 else []

    if node.at is not None:
        return get_path(
            node=node.at.previous_node,
            depth=depth + 1
        ) + [node.at.previous_node] + init_path
    else:
        return []

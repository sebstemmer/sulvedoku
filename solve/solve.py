from __future__ import annotations

import random
from typing import NamedTuple, Optional, Callable

from grid.grid import Coord, Cell, create_empty_grid, set_value_in_grid, \
    Grid


class At(NamedTuple):
    """
        Describes which node of the Sudoku solution path is handled.

        Attributes:
            coord (Coord):
                Coordinate of grid at which a value was set.
            tries (frozenset[int]):
                Already tried these values on other attempts.
            previous_node (SolutionPathNode):
                Previous node of solution path.
            is_trivial (int):
                Is the set value a trivial value (only one allowed value for this coordinate?).
    """
    coord: Coord
    tries: frozenset[int]
    previous_node: SolutionPathNode
    is_trivial: bool


class SolutionPathNode(NamedTuple):
    """
        Finding a solution of a Sudoku is modeled as a linked list. This list is called solution path.
        When a value is tried a new grid and hence a new node on the solution path is created.

        Attributes:
            grid (Grid):
                To each node on the solution path belongs a grid.
            at (Optional[At]):
                None at the start, afterwards describes the last set value at a specific coordinate.

    """
    grid: Grid
    at: Optional[At]


class MaxGoBackDepthReached(Exception):
    """
        Raised when the maximum go back depth in backtracking is reached.
    """
    pass


class GoBackFailed(Exception):
    """
        Raised when already at start and no previous node exists. This means there is no solution for the Sudoku.
    """
    pass


def solve_trivial_solutions(
        node: SolutionPathNode
) -> tuple[SolutionPathNode, bool] | None:
    """
        Find trivial solutions of a sudoku, creates and links the corresponding nodes on the path. Does not handle
        trivial solution opportunities that are created by setting trivial solution values.

        Args:
            node (SolutionPathNode):
                Node at which trivial solutions are found and solved.

        Returns:
            tuple[SolutionPathNode, bool] | None:
                None if Sudoku is in an invalid state after setting trivial solution values. Otherwise the current node
                 after solving trivial solutions and a flag if trivial solutions have been found.
    """
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

            new_node: SolutionPathNode = SolutionPathNode(
                grid=new_grid,
                at=At(
                    coord=coord,
                    tries=frozenset({allowed_values[0]}),
                    previous_node=previous_node,
                    is_trivial=True
                )
            )
            previous_node = new_node

    return previous_node, found_trivial_solutions


def recursively_solve_trivial_solutions(
        node: SolutionPathNode
) -> SolutionPathNode | None:
    """
        Find trivial solutions of a sudoku, creates and links the corresponding nodes on the path. Works recursively
        and handles trivial solution opportunities that are created by setting trivial solution values.

        Args:
            node (SolutionPathNode):
                Node at which trivial solutions are found and solved.

        Returns:
            SolutionPathNode | None:
                None if Sudoku is in an invalid state after setting trivial solution values. Otherwise the current node
                after solving trivial solutions.
    """
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


def try_other_value(
        coord: Coord,
        grid: Grid,
        already_tried: frozenset[int],
        node: SolutionPathNode,
        guess_strategy: Callable[[Grid], tuple[Coord, int]],
        max_go_back_depth: Optional[int]
) -> SolutionPathNode:
    """
        At a specific coordinate in the grid try a different value (that has not been tried before). If all values at
        this coordinate have been tried, go back to previous node (backtracking).

        Args:
            coord (Coord):
                Coordinate at which value is tried.
            grid (Grid):
                Current grid.
            already_tried (frozenset[int]):
                Already tried these values on other attempts.
            node (SolutionPathNode):
                Current node of solution path.
            guess_strategy (Callable[[Grid], tuple[Coord, int]]):
                Strategy used to determine which coordinate and value should be handled next.
            max_go_back_depth (int):
                Maximum depth of backtracking.

        Returns:
            SolutionPathNode:
                Final node of solution path because this function calls recursively_find_solution after trying a value.
    """
    allowed_values: tuple[int, ...] = grid.cells[coord].allowed_values

    possible_values: list[int] = [a for a in allowed_values if a not in already_tried]

    if len(possible_values) == 0:
        return go_back_to_previous_node_and_try_other_value(
            node=node,
            go_back_depth=0,
            guess_strategy=guess_strategy,
            max_go_back_depth=max_go_back_depth
        )

    next_try: int = random.choice(possible_values)

    next_try_grid: Grid | None = set_value_in_grid(
        grid=grid,
        coord=coord,
        value=next_try
    )

    new_already_tried: frozenset[int] = already_tried | frozenset({next_try})

    if next_try_grid is None:
        return try_other_value(
            coord=coord,
            grid=grid,
            already_tried=new_already_tried,
            node=node,
            guess_strategy=guess_strategy,
            max_go_back_depth=max_go_back_depth
        )

    next_try_node: SolutionPathNode = SolutionPathNode(
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
        guess_strategy=guess_strategy,
        max_go_back_depth=max_go_back_depth
    )


def go_back_to_previous_node_and_try_other_value(
        node: SolutionPathNode,
        guess_strategy: Callable[[Grid], tuple[Coord, int]],
        go_back_depth: int,
        max_go_back_depth: Optional[int]
) -> SolutionPathNode:
    """
        Go back to the previous node and try a different value (that has not been tried before). If all values at
        this coordinate have been tried, go back to previous node of previous node (backtracking).

        Args:
            node (SolutionPathNode):
                Node to go back from.
            guess_strategy (Callable[[Grid], tuple[Coord, int]]):
                 Strategy used to determine which coordinate and value should be handled next.
            go_back_depth (int):
                Current depth of backtracking.
            max_go_back_depth (int):
                Maximum depth of backtracking.

        Returns:
            SolutionPathNode:
                Final node of solution path.
    """
    if max_go_back_depth is not None and go_back_depth > max_go_back_depth:
        raise MaxGoBackDepthReached()

    at: At | None = node.at

    if node.at is None:
        raise GoBackFailed()

    if at.is_trivial:
        return go_back_to_previous_node_and_try_other_value(
            node=at.previous_node,
            guess_strategy=guess_strategy,
            go_back_depth=go_back_depth + 1,
            max_go_back_depth=max_go_back_depth
        )

    return try_other_value(
        coord=at.coord,
        grid=at.previous_node.grid,
        already_tried=at.tries,
        node=node.at.previous_node,
        guess_strategy=guess_strategy,
        max_go_back_depth=max_go_back_depth
    )


def random_guess_strategy(
        grid: Grid
) -> tuple[Coord, int]:
    """
        Randomly guess the next coordinate to handle.

        Args:
            grid (Grid):
                Current Sudoku grid.

        Returns:
            tuple[Coord, int]:
                Coordinate to handle and value to try.
    """
    coord = random.choice(grid.empty_coords)
    allowed_value = random.choice(grid.cells[coord].allowed_values)

    return coord, allowed_value


def ordered_guess_strategy(
        grid: Grid
) -> tuple[Coord, int]:
    """
    Take the next empty coordinate in row format to handle.

    Args:
        grid (Grid):
            Current Sudoku grid.

    Returns:
        tuple[Coord, int]:
            Coordinate to handle and value to try.
    """
    sorted_empty_coords = sorted(grid.empty_coords, key=lambda coord: coord.entry_idx)

    coord = sorted_empty_coords[0]
    allowed_value = random.choice(grid.cells[coord].allowed_values)

    return coord, allowed_value


def smallest_allowed_guess_strategy(
        grid: Grid
) -> tuple[Coord, int]:
    """
    Take the coordinate with the smallest amount of allowed values to handle next.

    Args:
        grid (Grid):
            Current Sudoku grid.

    Returns:
        tuple[Coord, int]:
            Coordinate to handle and value to try.
    """
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
        ],
        max_go_back_depth: Optional[int]
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
            guess_strategy=guess_strategy,
            go_back_depth=0,
            max_go_back_depth=max_go_back_depth
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

    already_tried: frozenset[int] = frozenset({next_coord_and_value[1]})

    if next_grid is None:
        return try_other_value(
            coord=next_coord_and_value[0],
            grid=handled_trivial_solutions.grid,
            already_tried=already_tried,
            node=handled_trivial_solutions,
            guess_strategy=guess_strategy,
            max_go_back_depth=max_go_back_depth
        )

    next_node: SolutionPathNode = SolutionPathNode(
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
        guess_strategy=guess_strategy,
        max_go_back_depth=max_go_back_depth
    )


def solve_grid(
        grid: Grid,
        guess_strategy: Callable[
            [Grid], tuple[Coord, int]
        ],
        max_go_back_depth: Optional[int]
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
    start: SolutionPathNode = SolutionPathNode(
        grid=grid,
        at=None
    )

    return recursively_find_solution(
        node=start,
        guess_strategy=guess_strategy,
        max_go_back_depth=max_go_back_depth
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
        guess_strategy=guess_strategy,
        max_go_back_depth=None,
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

    node: SolutionPathNode = SolutionPathNode(
        grid=grid,
        at=None
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
            guess_strategy=guess_strategy,
            max_go_back_depth=max_go_back_depth,
        )

    return final


def create_random_filled() -> Grid:
    return create_filled(
        max_go_back_depth=None,
        guess_strategy=random_guess_strategy
    ).grid


def get_path(node: SolutionPathNode, depth: int) -> list[SolutionPathNode]:
    init_path: list[SolutionPathNode] = [node] if depth == 0 else []

    if node.at is not None:
        return get_path(
            node=node.at.previous_node,
            depth=depth + 1
        ) + [node.at.previous_node] + init_path
    else:
        return []

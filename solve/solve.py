from __future__ import annotations

import random
from typing import NamedTuple, Optional, Callable

from grid.grid import Coord, create_empty_grid, set_value_in_grid, \
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
        Finding a solution of a Sudoku is modeled as a linked list (state graph). This list is called solution path.
        When a value is tried a new grid and hence a new node on the solution path is created.

        Attributes:
            grid (Grid):
                To each node on the solution path belongs a grid.
            at (Optional[At]):
                None at the start, afterwards describes the last set value at a specific coordinate.

    """
    grid: Grid
    at: Optional[At]


class MaxDepthReached(Exception):
    """
        Raised when the maximum depth in backtracking is reached.
    """
    pass


class GoBackFailed(Exception):
    """
        Raised when already at start and no previous node exists. This means there is no solution for the Sudoku.
    """
    pass


def get_new_depth(depth: int, max_depth: Optional[int]) -> int:
    """
        Increases the depth by one and raises MaxDepthReached if the maximum depth is defined and reached.

        Args:
            depth (int):
                Current depth.
            max_depth (Optional[int]):
                Maximum depth.

        Returns:
            int:
                New depth.

        Raises:
            MaxDepthReached:
                If the maximum depth is reached.
    """
    new_depth = depth + 1

    if max_depth is not None and new_depth > max_depth:
        raise MaxDepthReached()

    return new_depth


def solve_trivial_solutions(
        node: SolutionPathNode,
        depth: int,
        max_depth: Optional[int],
) -> tuple[tuple[SolutionPathNode, bool] | None, int]:
    """
        Find trivial solutions of a sudoku, creates and links the corresponding nodes on the path. Does not handle
        trivial solution opportunities that are created by setting trivial solution values.

        Args:
            node (SolutionPathNode):
                Node at which trivial solutions are found and solved.
            depth (int):
                Current depth.
            max_depth (Optional[int]):
                Maximum depth, None means no limit.

        Returns:
            tuple[tuple[SolutionPathNode, bool] | None, int]:
                The second element of the tuple is the depth. The first element of the tuple is None if Sudoku is in an
                invalid state after setting trivial solution values. Otherwise the first element is the current node after
                solving trivial solutions and a flag if trivial solutions have been found.

        Raises:
            MaxDepthReached:
                If the maximum depth is defined and reached.
    """
    found_trivial_solutions: bool = False

    previous_node = node
    new_depth = depth
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

            new_depth = get_new_depth(depth=new_depth, max_depth=max_depth)

            if new_grid is None:
                return None, new_depth

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

    return (previous_node, found_trivial_solutions), new_depth


def recursively_solve_trivial_solutions(
        node: SolutionPathNode,
        depth: int,
        max_depth: Optional[int] = None,
) -> tuple[SolutionPathNode | None, int]:
    """
        Find trivial solutions of a sudoku, creates and links the corresponding nodes on the path. Works recursively
        and handles trivial solution opportunities that are created by setting trivial solution values.

        Args:
            node (SolutionPathNode):
                Node at which trivial solutions are found and solved.
            depth (int):
                Current depth.
            max_depth (Optional[int]):
                Maximum depth, None means no limit.

        Returns:
            SolutionPathNode | None:
                The second element of the tuple is the depth. The first element is None if Sudoku is in an invalid state
                after setting trivial solution values. Otherwise the current node after solving trivial solutions.

        Raises:
            MaxDepthReached:
                If the maximum depth is defined and reached.
    """
    solved_trivial_solutions, new_depth = solve_trivial_solutions(
        node=node,
        depth=depth,
        max_depth=max_depth,
    )

    if solved_trivial_solutions is None:
        return None, new_depth

    after_find_trivial_solutions, trivial_solutions_solved = solved_trivial_solutions

    if not trivial_solutions_solved:
        return after_find_trivial_solutions, new_depth

    return recursively_solve_trivial_solutions(
        node=after_find_trivial_solutions,
        depth=new_depth,
        max_depth=max_depth,
    )


def try_other_value(
        coord: Coord,
        grid: Grid,
        already_tried: frozenset[int],
        depth: int,
        node: SolutionPathNode,
        guess_strategy: Callable[[Grid], tuple[Coord, int]],
        max_depth: Optional[int]
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
            depth (int):
                Current depth.
            guess_strategy (Callable[[Grid], tuple[Coord, int]]):
                Strategy used to determine which coordinate and value should be handled next.
            max_depth (int):
                Maximum depth. None means no limit.

        Returns:
            SolutionPathNode:
                Final node of solution path because this function calls recursively_find_solution after trying a value.

        Raises:
            MaxDepthReached:
                If the maximum depth is defined and reached.
    """
    allowed_values: tuple[int, ...] = grid.cells[coord].allowed_values

    possible_values: list[int] = [a for a in allowed_values if a not in already_tried]

    if len(possible_values) == 0:
        return go_back_to_previous_node_and_try_other_value(
            node=node,
            depth=depth,
            guess_strategy=guess_strategy,
            max_depth=max_depth
        )

    next_try: int = random.choice(possible_values)

    next_try_grid: Grid | None = set_value_in_grid(
        grid=grid,
        coord=coord,
        value=next_try
    )

    new_depth = get_new_depth(depth=depth, max_depth=max_depth)

    new_already_tried: frozenset[int] = already_tried | frozenset({next_try})

    if next_try_grid is None:
        return try_other_value(
            coord=coord,
            grid=grid,
            already_tried=new_already_tried,
            depth=new_depth,
            node=node,
            guess_strategy=guess_strategy,
            max_depth=max_depth
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
        depth=new_depth,
        guess_strategy=guess_strategy,
        max_depth=max_depth
    )


def go_back_to_previous_node_and_try_other_value(
        node: SolutionPathNode,
        depth: int,
        guess_strategy: Callable[[Grid], tuple[Coord, int]],
        max_depth: Optional[int]
) -> SolutionPathNode:
    """
        Go back to the previous node and try a different value (that has not been tried before). If all values at
        this coordinate have been tried, go back to previous node of previous node (backtracking).

        Args:
            node (SolutionPathNode):
                Node to go back from.
            depth (int):
                Current depth.
            guess_strategy (Callable[[Grid], tuple[Coord, int]]):
                 Strategy used to determine which coordinate and value should be handled next.
            max_depth (int):
                Maximum depth. None means no limit.

        Returns:
            SolutionPathNode:
                Final node of solution path.

        Raises:
            MaxDepthReached:
                If the maximum depth is defined and reached.
    """
    at: At | None = node.at

    if node.at is None:
        raise GoBackFailed()

    if at.is_trivial:
        return go_back_to_previous_node_and_try_other_value(
            node=at.previous_node,
            depth=depth,
            guess_strategy=guess_strategy,
            max_depth=max_depth
        )

    return try_other_value(
        coord=at.coord,
        grid=at.previous_node.grid,
        already_tried=at.tries,
        depth=depth,
        node=node.at.previous_node,
        guess_strategy=guess_strategy,
        max_depth=max_depth
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
        depth: int,
        guess_strategy: Callable[[Grid], tuple[Coord, int]] = smallest_allowed_guess_strategy,
        max_depth: Optional[int] = None
) -> SolutionPathNode:
    """
    Solve a Sudoku. This function is recursive.

    Args:
        node (SolutionPathNode):
            Node that contains grid to be solved.
        depth (int):
            Current depth.
        guess_strategy (Callable[[Grid], tuple[Coord, int]])
            Strategy used to determine which coordinate and value should be handled next.
        max_depth (Optional[int]):
            Maximum depth. None means no limit.

    Returns:
        SolutionPathNode:
            Last node of solution path containing the solution grid.

    Raises:
        MaxDepthReached:
            If the maximum depth is defined and reached.
        GoBackFailed:
            If no solution can be found.
    """

    handled_trivial_solutions, new_depth = recursively_solve_trivial_solutions(
        node=node,
        depth=depth,
        max_depth=max_depth
    )

    if handled_trivial_solutions is None:
        return go_back_to_previous_node_and_try_other_value(
            node=node,
            guess_strategy=guess_strategy,
            depth=new_depth,
            max_depth=max_depth
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

    new_depth = get_new_depth(depth=new_depth, max_depth=max_depth)

    if next_grid is None:
        return try_other_value(
            coord=next_coord_and_value[0],
            grid=handled_trivial_solutions.grid,
            already_tried=already_tried,
            depth=new_depth,
            node=handled_trivial_solutions,
            guess_strategy=guess_strategy,
            max_depth=max_depth
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
        depth=new_depth,
        guess_strategy=guess_strategy,
        max_depth=max_depth
    )


def solve_grid_until_no_trivial_solutions(
        grid: Grid
) -> SolutionPathNode:
    """
    Solve a grid until there are no trivial solutions left.

    Args:
        grid (Grid):
            Grid to solve.

    Returns:
        SolutionPathNode:
            Node when all trivial solutions are found.
    """

    node: SolutionPathNode = SolutionPathNode(
        grid=grid,
        at=None
    )

    solved_trivial_solutions, _ = recursively_solve_trivial_solutions(
        node=node,
        depth=0,
        max_depth=None
    )

    if solved_trivial_solutions is None:
        raise ValueError("grid is not valid")

    return solved_trivial_solutions


def solve_grid(
        grid: Grid,
        guess_strategy: Callable[
            [Grid], tuple[Coord, int]
        ] = smallest_allowed_guess_strategy,
        max_depth: Optional[int] = None
) -> Grid:
    """
    Solve a Sudoku grid.

    Args:
        grid (Grid):
            Grid to be solved.
        guess_strategy (Callable[[Grid], tuple[Coord, int]])
            Strategy used to determine which coordinate and value should be handled next.
        max_depth (Optional[int]):
            Maximum depth. None means no limit.

    Returns:
        Grid:
            Solution grid.

    Raises:
        MaxDepthReached:
            If the maximum depth is defined and reached.
        GoBackFailed:
            If no solution can be found.
    """
    start: SolutionPathNode = SolutionPathNode(
        grid=grid,
        at=None
    )

    return recursively_find_solution(
        node=start,
        depth=0,
        guess_strategy=guess_strategy,
        max_depth=max_depth
    ).grid


def create_filled(
        max_depth: Optional[int] = 0,
        guess_strategy: Callable[[Grid], tuple[Coord, int]] = random_guess_strategy
) -> Grid:
    """
    Create a completely filled Sudoku grid.

    Args:
        max_depth (Optional[int]):
            Maximum depth. None is no limit.
        guess_strategy (Callable[[Grid], tuple[Coord, int]]):
            Method for finding next coord and value to handle. Default is random because this results in maximum
            randomness in result grid.

    Returns:
        SolutionPathNode:
            Last node of solution path containing the filled grid.
    """
    empty_grid: Grid = create_empty_grid()

    final: Grid
    try:
        final = solve_grid(
            grid=empty_grid,
            max_depth=max_depth,
            guess_strategy=guess_strategy
        )
    except (GoBackFailed, MaxDepthReached):
        final = create_filled(
            max_depth=max_depth,
            guess_strategy=guess_strategy
        )

    return final

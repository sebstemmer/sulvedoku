from __future__ import annotations

import random
from enum import Enum
from typing import NamedTuple, Optional

from grid import Coord, Cell, create_empty_grid, get_random_empty_where_allowed_values_is_len_1, set_value_in_grid, \
    remove_values_from_grid, all_coords_0_to_80


class At(NamedTuple):
    coord: Coord
    value_tries: list[int]
    previous_node: SolutionPathNode
    is_trivial: bool


class SolutionPathNode(NamedTuple):
    grid: dict[Coord, Cell]
    at: Optional[At]
    recursion_depth: int
    max_go_back_depth: Optional[int]
    method: FillPathCreationMethod


class RemoveCluesResult(NamedTuple):
    grid: dict[Coord, Cell]
    depth: int


class FillPathCreationMethod(Enum):
    ORDERED = 1
    RANDOM = 2


class MaxGoBackDepthReached(Exception):
    pass


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


def create_node(
        node: SolutionPathNode,
        grid: dict[Coord, Cell],
        at: Optional[At],
        recursion_depth: int
) -> SolutionPathNode:
    return SolutionPathNode(
        grid=grid,
        at=at,
        recursion_depth=recursion_depth,
        max_go_back_depth=node.max_go_back_depth,
        method=node.method,
    )


def create_start_node(
        grid: dict[Coord, Cell],
        max_go_back_depth: Optional[int],
        method: FillPathCreationMethod
) -> SolutionPathNode:
    return SolutionPathNode(
        grid=grid,
        at=None,
        recursion_depth=0,
        max_go_back_depth=max_go_back_depth,
        method=method
    )


def find_trivial_solutions(
        node: SolutionPathNode,
        recursion_depth: int,
) -> SolutionPathNode:
    """
        Recursively find trivial solutions, if no more trivial solutions are found, return up-to-date SolutionPathNode.

        Args:
            node (SolutionPathNode): SolutionPathNode before finding trivial solutions.
            recursion_depth (int): Current depth of recursion.

        Returns:
            SolutionPathNode: Node when all trivial solutions are found.
    """
    trivial_solution: Optional[
        tuple[Coord, Cell]
    ] = get_random_empty_where_allowed_values_is_len_1(
        grid=node.grid
    )

    if trivial_solution is None:
        return node

    coord = trivial_solution[0]
    value = trivial_solution[1].allowed_values[0]

    new_grid: dict[Coord, Cell] = set_value_in_grid(
        grid=node.grid,
        coord=coord,
        value=value
    )

    new_node: SolutionPathNode = create_node(
        node=node,
        grid=new_grid,
        at=At(
            coord=coord,
            value_tries=[value],
            previous_node=node,
            is_trivial=True
        ),
        recursion_depth=recursion_depth,
    )

    return find_trivial_solutions(
        node=new_node,
        recursion_depth=recursion_depth + 1
    )


def go_back_to_previous_node_and_try_other_value(
        node: SolutionPathNode,
        recursion_depth: int,
        go_back_depth: int,
) -> SolutionPathNode:
    """
        Backtracking: Goes back to first previous node with still values to try out
        and continues trying out a new value.

        Args:
            node (SolutionPathNode): Node to go back from.
            recursion_depth (int): Current depth of recursion.
            go_back_depth (int): Current depth of backtracking.

        Returns:
            SolutionPathNode: Last node of solution path containing the solution grid.

        Raises:
            MaxGoBackDepthReached: If maximum backtracking depth is reached
            GoBackFailed: If no solution can be found.
    """
    if node.max_go_back_depth is not None and go_back_depth > node.max_go_back_depth:
        raise MaxGoBackDepthReached()

    at: Optional[At] = node.at

    if at is None:
        raise GoBackFailed()

    allowed_values: list[int] = at.previous_node.grid[
        at.coord
    ].allowed_values

    not_yet_tried: list[int] = [
        v for v in allowed_values if v not in at.value_tries
    ]

    if len(not_yet_tried) == 0:
        return go_back_to_previous_node_and_try_other_value(
            node=at.previous_node,
            recursion_depth=recursion_depth + 1,
            go_back_depth=go_back_depth + 1,
        )

    random_idx: int = 0 if len(not_yet_tried) == 1 else random.randint(
        0, len(not_yet_tried) - 1
    )

    other_value_try: int = not_yet_tried[random_idx]

    other_value_try_grid: dict[Coord, Cell] = set_value_in_grid(
        grid=at.previous_node.grid,
        coord=at.coord,
        value=other_value_try
    )

    other_value_try_node: SolutionPathNode = create_node(
        node=node,
        grid=other_value_try_grid,
        at=At(
            coord=at.coord,
            value_tries=at.value_tries.copy() + [other_value_try],
            previous_node=at.previous_node,
            is_trivial=False
        ),
        recursion_depth=recursion_depth
    )

    return recursively_find_solution(
        node=other_value_try_node,
        recursion_depth=recursion_depth + 1
    )


def find_next_coord_to_handle_for_ordered(
        grid: dict[Coord, Cell]
) -> Optional[Coord]:
    for coord in all_coords_0_to_80:
        if grid[coord].value == 0:
            return coord

    return None


def find_next_coord_to_handle_for_random(
        grid: dict[Coord, Cell]
) -> Optional[Coord]:
    all_coords_0_to_80_list: list[Coord] = list(all_coords_0_to_80)
    random.shuffle(all_coords_0_to_80_list)
    for coord in all_coords_0_to_80_list:
        if grid[coord].value == 0:
            return coord

    return None


def find_next_coord_to_handle(
        node: SolutionPathNode
) -> Optional[Coord]:
    match node.method:
        case FillPathCreationMethod.ORDERED:
            return find_next_coord_to_handle_for_ordered(grid=node.grid)
        case FillPathCreationMethod.RANDOM:
            return find_next_coord_to_handle_for_random(grid=node.grid)


def recursively_find_solution(
        node: SolutionPathNode,
        recursion_depth: int
) -> SolutionPathNode:
    """
    Recursive function to solve a (not necessary valid) grid.

    Args:
        node (SolutionPathNode): Node that contains the grid and the other parameters to solve.
        recursion_depth (int): Current depth of recursion.

    Returns:
        SolutionPathNode: Last node of solution path containing the solution grid.

    Raises:
        MaxGoBackDepthReached: If maximum backtracking depth is reached
        GoBackFailed: If no solution can be found.
    """
    handled_trivial_solutions: SolutionPathNode = find_trivial_solutions(
        node=node,
        recursion_depth=recursion_depth + 1
    )

    next_coord: Optional[Coord] = find_next_coord_to_handle(
        handled_trivial_solutions
    )

    if next_coord is None:
        return handled_trivial_solutions

    allowed_values: list[int] = handled_trivial_solutions.grid[next_coord].allowed_values

    if len(allowed_values) == 0:
        return go_back_to_previous_node_and_try_other_value(
            node=handled_trivial_solutions,
            depth=recursion_depth + 1,
            go_back_depth=0,
        )

    random_idx: int = 0 if len(allowed_values) == 1 else random.randint(
        0, len(allowed_values) - 1
    )

    next_value: int = allowed_values[random_idx]

    next_grid: dict[Coord, Cell] = set_value_in_grid(
        grid=handled_trivial_solutions.grid,
        coord=next_coord,
        value=next_value
    )

    next_node: SolutionPathNode = create_node(
        node=handled_trivial_solutions,
        grid=next_grid,
        at=At(
            coord=next_coord,
            value_tries=[next_value],
            previous_node=handled_trivial_solutions,
            is_trivial=False
        ),
        recursion_depth=recursion_depth
    )

    return recursively_find_solution(
        node=next_node,
        recursion_depth=recursion_depth + 1,
    )


def solve_grid(
        grid: dict[Coord, Cell],
        max_go_back_depth: Optional[int],
        method: FillPathCreationMethod
) -> SolutionPathNode:
    """
    Solve a (not necessary valid) grid.

    Args:
        grid (dict[Coord, Cell]): Grid to solve.
        max_go_back_depth (Optional[int]): Maximum depth of going back when backtracking (None means no maximum depth).
        method (FillPathCreationMethod): Method used for solving.

    Returns:
        SolutionPathNode: Last node of solution path containing the solution grid.

    Raises:
        MaxGoBackDepthReached: If maximum backtracking depth is reached
        GoBackFailed: If no solution can be found.
    """
    start: SolutionPathNode = create_start_node(
        grid=grid,
        max_go_back_depth=max_go_back_depth,
        method=method
    )

    return recursively_find_solution(
        node=start,
        recursion_depth=0
    )


def solve_valid_grid(
        grid: dict[Coord, Cell],
        method: FillPathCreationMethod,
) -> SolutionPathNode:
    """
    Solve a VALID grid.

    Args:
        grid (dict[Coord, Cell]): Grid to solve.
        method (FillPathCreationMethod): Method used for solving.

    Returns:
        SolutionPathNode: Last node of solution path containing the solution grid.
    """
    return solve_grid(
        grid=grid,
        max_go_back_depth=None,
        method=method
    )


def solve_valid_grid_until_no_trivial_solutions(
        grid: dict[Coord, Cell]
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
        max_go_back_depth=None,
        method=FillPathCreationMethod.ORDERED
    )

    return find_trivial_solutions(
        node=node,
        recursion_depth=0,
    )


def create_filled(
        max_go_back_depth: Optional[int],
        method: FillPathCreationMethod
) -> SolutionPathNode:
    """
    Create a filled grid.

    Args:
        max_go_back_depth (Optional[int]): Maximum depth of going back when backtracking (None means no maximum depth).
        method (FillPathCreationMethod): Method used for creation.

    Returns:
        SolutionPathNode: Last node of solution path containing the filled grid.
    """
    empty_grid: dict[Coord, Cell] = create_empty_grid()

    try:
        final: SolutionPathNode = solve_grid(
            grid=empty_grid,
            max_go_back_depth=max_go_back_depth,
            method=method,
        )
    except (GoBackFailed, MaxGoBackDepthReached):
        final: SolutionPathNode = create_filled(
            max_go_back_depth=max_go_back_depth,
            method=method
        )

    return final


# todo for sebstemmer: checken ob hier nicht grid rein muss
# hängt an max go back length von node das ist falsch
def check_if_has_unique_solution(
        node: SolutionPathNode,
        solution_grid: dict[Coord, Cell]
) -> bool:
    if len(node.fill_path) == 0:
        return True

    coord: Coord = node.fill_path[0]

    solution_value: int = solution_grid[coord].value

    allowed_values_without_solution_value: list[int] = [
        v for v in node.grid[coord].allowed_values if v != solution_value
    ]

    fill_path_tail: list[Coord] = [c for c in node.fill_path[1:]]

    if len(allowed_values_without_solution_value) > 0:
        for value in allowed_values_without_solution_value:
            new_grid = set_value_in_grid(
                grid=node.grid,
                coord=coord,
                value=value
            )

            start: SolutionPathNode = create_start_node(
                grid=new_grid,
                fill_path=fill_path_tail,
                max_go_back_depth=None
            )

            try:
                _: SolutionPathNode = recursively_find_solution(
                    node=start,
                    depth=0
                )
                return False
            except:
                pass

    next_grid: dict[Coord, Cell] = set_value_in_grid(
        grid=node.grid,
        coord=coord,
        value=solution_value
    )

    next_node: SolutionPathNode = create_start_node(
        grid=next_grid,
        fill_path=fill_path_tail,
        max_go_back_depth=None
    )

    return check_if_has_unique_solution(
        node=next_node,
        solution_grid=solution_grid
    )


def check_if_has_unique_solution_from_grid(
        grid: dict[Coord, Cell],
        solution_grid: dict[Coord, Cell]
) -> bool:
    return check_if_has_unique_solution(
        node=SolutionPathNode(
            grid=grid,
            at=None,
            recursion_depth=0,
            max_go_back_depth=None,
            method=FillPathCreationMethod.ORDERED
        ),
        solution_grid=solution_grid
    )


def create_partially_filled(
        filled_grid: dict[Coord, Cell],
        num_empties: int
) -> SolutionPathNode:
    fill_path: list[Coord] = random.sample(all_coords_0_to_80, num_empties)

    grid: dict[Coord, Cell] = remove_values_from_grid(
        grid=filled_grid,
        coords=fill_path
    )

    partially_filled: SolutionPathNode = create_start_node(
        grid=grid,
        fill_path=fill_path,
        max_go_back_depth=None
    )

    has_unique_solution: bool = check_if_has_unique_solution(
        node=partially_filled,
        solution_grid=filled_grid
    )

    if has_unique_solution:
        return partially_filled
    else:
        print("restart")
        return create_partially_filled(
            filled_grid=filled_grid,
            num_empties=num_empties
        )


def get_path(node: SolutionPathNode, depth: int) -> list[SolutionPathNode]:
    init_path: list[SolutionPathNode] = [node] if depth == 0 else []

    if node.at is not None:
        return get_path(
            node=node.at.previous_node,
            depth=depth + 1
        ) + [node.at.previous_node] + init_path
    else:
        return []


def recursively_remove_clues(
        grid: dict[Coord, Cell],
        num_remaining_clues: int,
        clues: list[Coord],
        solution_grid: dict[Coord, Cell],
        grid_to_remove_threshold: int,
        grid_to_remove: list[tuple[dict[Coord, Cell], Coord]],
        depth: int,
        max_depth: int
) -> RemoveCluesResult:
    if depth > max_depth:
        raise MaxRemoveCluesDepthReached(
            grid_to_remove=grid_to_remove
        )

    print(f"num clues: {len(clues)}")
    print(f"depth: {depth}")
    if len(clues) == num_remaining_clues:
        return RemoveCluesResult(
            grid=grid,
            depth=depth
        )

    shuffled_clues: list[Coord] = list(clues)
    random.shuffle(shuffled_clues)

    for_depth: int = depth

    for remove in shuffled_clues:
        # todo nur ein value removen
        grid_after_removing: dict[Coord, Cell] = remove_values_from_grid(
            grid=grid,
            coords=[remove]
        )

        node_after_removing: SolutionPathNode = create_start_node(
            grid=grid_after_removing,
            max_go_back_depth=None,
            method=FillPathCreationMethod.RANDOM,
        )

        has_unique_solution: bool = check_if_has_unique_solution(
            node=node_after_removing,
            solution_grid=solution_grid,
        )

        if has_unique_solution:
            if len(clues) < grid_to_remove_threshold:
                grid_to_remove.append(
                    (
                        grid_after_removing,
                        remove
                    )
                )

            try:
                result: RemoveCluesResult = recursively_remove_clues(
                    grid=grid_after_removing,
                    num_remaining_clues=num_remaining_clues,
                    clues=[c for c in clues if c != remove],
                    solution_grid=solution_grid,
                    grid_to_remove_threshold=grid_to_remove_threshold,
                    grid_to_remove=grid_to_remove,
                    depth=for_depth + 1,
                    max_depth=max_depth
                )
                return result
            except SolutionNotUnique as s:
                for_depth = s.depth
                pass
        else:
            raise SolutionNotUnique(depth=for_depth)

    raise SolutionNotUnique(depth=for_depth)


def create_partially_filled_from_creation_path(
        num_clues: int,
        grid_to_remove_threshold: int,
        grid_to_remove: list[tuple[dict[Coord, Cell], Coord]],
        max_depth: int
) -> RemoveCluesResult:
    filled: SolutionPathNode = create_filled(
        method=FillPathCreationMethod.RANDOM,
        max_go_back_depth=-1
    )

    return recursively_remove_clues(
        grid=filled.grid,
        num_remaining_clues=num_clues,
        clues=list(all_coords_0_to_80),
        solution_grid=filled.grid,
        grid_to_remove_threshold=grid_to_remove_threshold,
        grid_to_remove=grid_to_remove,
        depth=0,
        max_depth=max_depth
    )

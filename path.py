from __future__ import annotations
from typing import NamedTuple, Optional
import random
from grid import Coord, Cell, create_empty_grid, get_random_empty_where_allowed_values_is_len_1, set_value_in_grid, remove_values_from_grid
from enum import Enum


class At(NamedTuple):
    coord: Coord
    value_tries: list[int]
    fill_path_idx: Optional[int]
    previous_node: SolutionPathNode
    is_trivial: bool


class SolutionPathNode(NamedTuple):
    grid:  dict[Coord, Cell]
    fill_path: list[Coord]
    at: Optional[At]
    depth: int
    max_go_back_depth: Optional[int]


class FillPathCreationMethod(Enum):
    ORDERED = 1
    RANDOM = 2


class MaxGoBackDepthReached(Exception):
    pass


class GoBackFailed(Exception):
    pass


def create_fill_path(
        grid: dict[Coord, Cell],
        method: FillPathCreationMethod
) -> list[Coord]:
    coords: list[Coord] = [
        coord
        for row_idx in range(9)
        for col_idx in range(9)
        for coord in [Coord(row_idx=row_idx, col_idx=col_idx)]
        if grid[coord].value == 0
    ]

    match method:
        case FillPathCreationMethod.ORDERED:
            return coords
        case FillPathCreationMethod.RANDOM:
            random.shuffle(coords)
            return coords


def create_start_node(
    grid: dict[Coord, Cell],
    fill_path: list[Coord],
    max_go_back_depth: Optional[int]
) -> SolutionPathNode:
    return SolutionPathNode(
        grid=grid,
        fill_path=fill_path,
        at=None,
        depth=0,
        max_go_back_depth=max_go_back_depth
    )


def find_trivial_solutions(
    node: SolutionPathNode,
    depth: int,
    coord_to_all_coords_in_row_col_or_square: dict[
        Coord, set[Coord]
    ]
) -> SolutionPathNode:
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
        value=value,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )

    fill_path_idx: Optional[int] = None if node.at is None else node.at.fill_path_idx

    new_path: SolutionPathNode = SolutionPathNode(
        grid=new_grid,
        fill_path=node.fill_path,
        at=At(
            coord=coord,
            value_tries=[value],
            fill_path_idx=fill_path_idx,
            previous_node=node,
            is_trivial=True
        ),
        depth=depth,
        max_go_back_depth=node.max_go_back_depth
    )

    return find_trivial_solutions(
        node=new_path,
        depth=depth + 1,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )


def find_next_fill_path_idx(
    node: SolutionPathNode
) -> Optional[int]:
    start: int
    if node.at is None or node.at.fill_path_idx is None:
        start = 0
    else:
        start = node.at.fill_path_idx + 1

    for idx in range(start, len(node.fill_path)):
        coord: Coord = node.fill_path[idx]

        if node.grid[coord].value == 0:
            return idx

    return None


def go_back_to_previous_node_and_try_other_value(
        node: SolutionPathNode,
        depth: int,
        go_back_depth: int,
        coord_to_all_coords_in_row_col_or_square: dict[
        Coord, set[Coord]
        ]
) -> SolutionPathNode:
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
            at.previous_node,
            depth=depth + 1,
            go_back_depth=go_back_depth + 1,
            coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
        )

    random_idx: int = 0 if len(not_yet_tried) == 1 else random.randint(
        0, len(not_yet_tried)-1
    )

    other_value_try: int = not_yet_tried[random_idx]

    other_value_try_grid: dict[Coord, Cell] = set_value_in_grid(
        grid=at.previous_node.grid,
        coord=at.coord,
        value=other_value_try,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )

    other_value_try_node: SolutionPathNode = SolutionPathNode(
        grid=other_value_try_grid,
        fill_path=node.fill_path,
        at=At(
            coord=at.coord,
            value_tries=at.value_tries.copy() + [other_value_try],
            fill_path_idx=at.fill_path_idx,
            previous_node=at.previous_node,
            is_trivial=False
        ),
        depth=depth,
        max_go_back_depth=node.max_go_back_depth,
    )

    return recursively_find_solution(
        node=other_value_try_node,
        depth=depth + 1,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )


def recursively_find_solution(
    node: SolutionPathNode,
    depth: int,
    coord_to_all_coords_in_row_col_or_square: dict[
        Coord, set[Coord]
    ]
) -> SolutionPathNode:
    handled_trivial_solutions: SolutionPathNode = find_trivial_solutions(
        node=node,
        depth=depth + 1,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )

    next_fill_path_idx: Optional[int] = find_next_fill_path_idx(
        node=handled_trivial_solutions
    )

    if next_fill_path_idx is None:
        return handled_trivial_solutions

    next_coord: Coord = handled_trivial_solutions.fill_path[next_fill_path_idx]

    allowed_values: list[int] = handled_trivial_solutions.grid[next_coord].allowed_values

    if len(allowed_values) == 0:
        return go_back_to_previous_node_and_try_other_value(
            node=handled_trivial_solutions,
            depth=depth + 1,
            go_back_depth=0,
            coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
        )

    random_idx: int = 0 if len(allowed_values) == 1 else random.randint(
        0, len(allowed_values)-1
    )

    next_value: int = allowed_values[random_idx]

    next_grid: dict[Coord, Cell] = set_value_in_grid(
        grid=handled_trivial_solutions.grid,
        coord=next_coord,
        value=next_value,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )

    next_node: SolutionPathNode = SolutionPathNode(
        grid=next_grid,
        fill_path=handled_trivial_solutions.fill_path,
        at=At(
            coord=next_coord,
            value_tries=[next_value],
            fill_path_idx=next_fill_path_idx,
            previous_node=handled_trivial_solutions,
            is_trivial=False
        ),
        depth=depth,
        max_go_back_depth=handled_trivial_solutions.max_go_back_depth
    )

    return recursively_find_solution(
        node=next_node,
        depth=depth + 1,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )


def create_filled(
    method: FillPathCreationMethod,
    coord_to_all_coords_in_row_col_or_square: dict[
        Coord, set[Coord]
    ],
    max_go_back_depth: int
) -> SolutionPathNode:
    empty_grid: dict[Coord, Cell] = create_empty_grid(
        all_coords=list(coord_to_all_coords_in_row_col_or_square.keys())
    )

    fill_path: list[Coord] = create_fill_path(
        grid=empty_grid, method=FillPathCreationMethod.ORDERED
    )

    start: SolutionPathNode = create_start_node(
        grid=empty_grid,
        fill_path=fill_path,
        max_go_back_depth=max_go_back_depth
    )

    try:
        final: SolutionPathNode = recursively_find_solution(
            node=start,
            depth=0,
            coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
        )
    except (GoBackFailed, MaxGoBackDepthReached):
        final: SolutionPathNode = create_filled(
            method=method,
            coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square,
            max_go_back_depth=max_go_back_depth
        )

    return final


def check_if_has_unique_solution(
        node: SolutionPathNode,
        solution: dict[Coord, Cell],
        coord_to_all_coords_in_row_col_or_square: dict[
        Coord, set[Coord]
        ],
) -> bool:
    if len(node.fill_path) == 0:
        return True

    coord: Coord = node.fill_path[0]

    solution_value: int = solution[coord].value

    allowed_values_without_solution_value: list[int] = [
        v for v in node.grid[coord].allowed_values if v != solution_value
    ]

    fill_path_tail: list[Coord] = [c for c in node.fill_path[1:]]

    if len(allowed_values_without_solution_value) > 0:
        for value in allowed_values_without_solution_value:
            new_grid = set_value_in_grid(
                grid=node.grid,
                coord=coord,
                value=value,
                coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
            )

            start: SolutionPathNode = create_start_node(
                grid=new_grid,
                fill_path=fill_path_tail,
                max_go_back_depth=None
            )

            try:
                _: SolutionPathNode = recursively_find_solution(
                    node=start,
                    depth=0,
                    coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
                )
                return False
            except:
                pass

    next_grid: dict[Coord, Cell] = set_value_in_grid(
        grid=node.grid,
        coord=coord,
        value=solution_value,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )

    next: SolutionPathNode = create_start_node(
        grid=next_grid,
        fill_path=fill_path_tail,
        max_go_back_depth=None
    )

    return check_if_has_unique_solution(
        node=next,
        solution=solution,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )


def create_partially_filled(
    filled: dict[Coord, Cell],
    num_empties: int,
    all_coords: list[Coord],
    coord_to_all_coords_in_row_col_or_square: dict[
        Coord, set[Coord]
    ],
) -> SolutionPathNode:
    fill_path: list[Coord] = random.sample(all_coords, num_empties)

    grid: dict[Coord, Cell] = remove_values_from_grid(
        grid=filled,
        coords=fill_path,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )

    partially_filled: SolutionPathNode = create_start_node(
        grid=grid,
        fill_path=fill_path,
        max_go_back_depth=None
    )

    has_unique_solution: bool = check_if_has_unique_solution(
        node=partially_filled,
        solution=filled,
        coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
    )

    if has_unique_solution:
        return partially_filled
    else:
        print("restart")
        return create_partially_filled(
            filled=filled,
            num_empties=num_empties,
            all_coords=all_coords,
            coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
        )

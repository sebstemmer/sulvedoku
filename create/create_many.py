from solve.path import create_filled, random_guess_strategy
from grid.grid import grid_to_str, all_coords_0_to_80
from create.create_core import recursively_remove_values, RemovePathNode
import sys

sys.setrecursionlimit(int(1e9))


def get_path(node: RemovePathNode, depth: int) -> list[RemovePathNode]:
    init_path: list[RemovePathNode] = [node] if depth == 0 else []

    if node.at is not None:
        return get_path(
            node=node.at.previous_node,
            depth=depth + 1
        ) + [node.at.previous_node] + init_path
    else:
        return []

filled = create_filled(max_go_back_depth=-1, guess_strategy=random_guess_strategy)

twenty_five = recursively_remove_values(
    node=RemovePathNode(
        grid=filled.grid,
        filled_coords=all_coords_0_to_80,
        at=None,
        selection_depth=0
    ),
    solution_grid=filled.grid,
    num_filled_target=22,
    max_selection_depth=int(1e4)
)

print(grid_to_str(twenty_five.grid, "\n", " "))

r = recursively_remove_values(
    node=RemovePathNode(
        grid=twenty_five.grid,
        filled_coords=tuple([c for c in all_coords_0_to_80 if twenty_five.grid.cells[c] != 0]),
        at=None,
        selection_depth=0
    ),
    solution_grid=filled.grid,
    num_filled_target=21,
    max_selection_depth=int(1e5)
)

#result = create_grid(
#    num_filled_target=21,
#    max_selection_depth=int(1e3)
#)

#print([p.selection_depth for p in get_path(result, 0)])

print(grid_to_str(r.grid, "\n", " "))

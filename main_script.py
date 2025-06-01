import path
import grid

coord_to_all_coords_in_row_col_or_square: dict[
    grid.Coord, set[grid.Coord]
] = grid.create_coord_to_all_coords_in_row_col_or_square()
all_coords: list[grid.Coord] = list(
    coord_to_all_coords_in_row_col_or_square.keys()
)


filled: path.SolutionPathNode = path.create_filled(
    method=path.FillPathCreationMethod.RANDOM,
    coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square,
    max_go_back_depth=-1
)

print("filled created")

partially_filled: path.SolutionPathNode = path.create_partially_filled(
    filled=filled.grid,
    num_empties=50,
    all_coords=all_coords,
    coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
)

print(grid.is_valid(
    grid=partially_filled.grid,
      coord_to_all_coords_in_row_col_or_square=coord_to_all_coords_in_row_col_or_square
      ))

print(
    grid.grid_to_str(
        grid=partially_filled.grid,
        row_join="\n",
        col_join=" "
    )
)

import numpy as np
import torch

from grid.grid import Coord, all_coords_0_to_80, Grid
from neural_solve.get_most_certain_coord_and_index_value import get_most_certain_coord_and_index_value
from neural_solve.train_utils import create_sudoku_net_model, final_model_path


def create_x_target_mask_and_allowed_values_mask(
        coord_to_value: dict[Coord, int],
        coord_to_allowed_values: dict[Coord, tuple[int, ...]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.float32
    x_81x19 = torch.zeros(81 * 19, dtype=dtype)
    target_mask_81 = torch.zeros(81, dtype=dtype)
    allowed_mask_81x9 = torch.zeros(81 * 9, dtype=dtype)

    allowed_value_indexes = np.fromiter(
        (
            a - 1 + (entry_idx * 9)
            for entry_idx, coord in enumerate(all_coords_0_to_80)
            for a in coord_to_allowed_values[coord]
        ),
        dtype=np.uint16,
    )

    x_indices_aws = np.fromiter(
        (
            a + 9 + (entry_idx * 19)
            for entry_idx, coord in enumerate(all_coords_0_to_80)
            for a in coord_to_allowed_values[coord]
        ),
        dtype=np.uint16,
    )

    allowed_mask_81x9[allowed_value_indexes] = 1.0

    x_indices_values = np.fromiter(
        [coord_to_value[coord] + (entry_idx * 19) for (entry_idx, coord) in enumerate(all_coords_0_to_80)],
        dtype=np.uint16
    )

    x_81x19[np.concatenate([x_indices_values, x_indices_aws])] = 1.0

    target_mask_81[[coord_to_value[coord] == 0 for coord in all_coords_0_to_80]] = 1.0

    x_19_81 = x_81x19.reshape(81, 19).transpose(0, 1)
    allowed_mask_81_9 = allowed_mask_81x9.reshape(81, 9)

    return x_19_81.unsqueeze(0), target_mask_81.unsqueeze(0), allowed_mask_81_9.unsqueeze(0)


model = create_sudoku_net_model()
model.eval()
model.load_state_dict(torch.load(final_model_path))


def neural_guess_strategy(
        grid: Grid
) -> tuple[Coord, int]:
    coord_to_value: dict[Coord, int] = {}
    coord_to_allowed_values: dict[Coord, tuple[int, ...]] = {}

    for coord in all_coords_0_to_80:
        cell = grid.cells[coord]
        coord_to_value[coord] = cell.value
        coord_to_allowed_values[coord] = cell.allowed_values

    x_n_19_81, target_mask_n_81, allowed_values_mask_n_81_9 = create_x_target_mask_and_allowed_values_mask(
        coord_to_value=coord_to_value,
        coord_to_allowed_values=coord_to_allowed_values
    )

    with torch.inference_mode():
        logits_n_81_9 = model(x_n_19_81)

    most_certain_coord_and_index_value = get_most_certain_coord_and_index_value(
        allowed_mask_n_81_9=allowed_values_mask_n_81_9,
        logits_n_81_9=logits_n_81_9,
        target_mask_n_81=target_mask_n_81
    )

    return most_certain_coord_and_index_value[0][0], most_certain_coord_and_index_value[1][0] + 1

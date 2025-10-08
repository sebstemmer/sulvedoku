import torch
import torch.nn.functional as F

from grid.grid import Coord, all_coords_0_to_80


def get_most_certain_coord_and_index_value(
        allowed_mask_n_81_9: torch.Tensor,
        logits_n_81_9: torch.Tensor,
        target_mask_n_81: torch.Tensor
) -> tuple[list[Coord], list[int]]:
    """
        For each sample in the batch calculates the coordinate and indexed value where the neural network is the most
        certain. Excludes values that are not allowed by the Sudoku constraints by setting there probability via
        softmax to 0.

        Args:
            allowed_mask_n_81_9 (torch.Tensor):
                N x 81 x 9: Indicates for each cell which values are allowed by the Sudoku constraints. Values are index (0 - 8)
                and range from 1.0 if allowed, 0.0 if not.
            logits_n_81_9 (torch.Tensor):
                N x 81 x 9: Result of the neural network. Indicates the probability of each value for each cell.
            target_mask_n_81 (torch.Tensor):
                81: Only empty cells are considered, this mask indicates which cells are empty (1.0) and which are
                filled (0.0).
            big_number (float):
                Big number to set to softmax non-allowed values to 0.


        Returns:
            tuple[list[Coord], list[int]]:
                For each sample in the batch coordinate and index value (0 - 8) where the neural network is the most certain.
    """
    logits_non_allowed_to_inf_n_81_9 = torch.where(
        condition=allowed_mask_n_81_9 > 0.5,
        input=logits_n_81_9,
        other=-1e9
    )

    probs_n_81_9 = F.softmax(logits_non_allowed_to_inf_n_81_9, dim=2) * target_mask_n_81.unsqueeze(2)

    _, argmax_idx_n = probs_n_81_9.reshape(probs_n_81_9.shape[0], -1).max(dim=1)

    argmax_entry_idx = argmax_idx_n // 9

    most_certain_coord: list[Coord] = [all_coords_0_to_80[i] for i in argmax_entry_idx]
    most_certain_indexed_value: list[int] = (argmax_idx_n % 9).to(torch.uint8).tolist()

    return most_certain_coord, most_certain_indexed_value

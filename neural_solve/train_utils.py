import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from neural_solve.model import SudokuNet
from common.log_info import log_info
from neural_solve.get_most_certain_coord_and_index_value import get_most_certain_coord_and_index_value

train_data_path: str = "./data/neural-solver-train-data.csv"

batch_size: int = 1024

learning_rate: float = 1e-4
weight_decay: float = 1e-3

load_from_epoch: int | None = None
epoch_key: str = "epoch"
model_state_key: str = "model_state"
optimizer_state_key: str = "optimizer_state"
train_loss_key: str = "train_loss"
valid_loss_key: str = "valid_loss"
valid_success_rate_key: str = "valid_success_rate"

feature_maps: int = 456
row_col_box_out_features: int = 27
dense_out_features: int = 729

num_epochs: int = 100

training_device = torch.device("mps")

final_model_path: str = "./data/final_sudoku_net_model.pth"


def get_model_weight_path(epoch: int) -> str:
    return f"./data/sudoku_net_training_checkpoints/{epoch}.pth"


def create_sudoku_net_model() -> SudokuNet:
    return SudokuNet(
        feature_maps=feature_maps,
        row_col_box_out_features=row_col_box_out_features,
        dense_out_features=dense_out_features
    )


def masked_ce_loss(
        logits_n_81_9: torch.Tensor,
        targets_n_81: torch.Tensor,
        mask_n_81: torch.Tensor
) -> torch.Tensor:
    logits_nx81_9 = logits_n_81_9.reshape(-1, 9)
    targets_nx81 = targets_n_81.reshape(-1)
    mask_nx81 = mask_n_81.reshape(-1)

    losses_nx81 = F.cross_entropy(
        input=logits_nx81_9,
        target=targets_nx81,
        reduction="none"
    )

    return (losses_nx81 * mask_nx81).sum() / mask_nx81.sum()


def validate(
        neural_network: SudokuNet,
        valid_loader: DataLoader,
        device: torch.device
) -> tuple[float, float]:
    neural_network.eval()

    total_average_batch_loss: float = 0.0

    success_rate = 0.0

    with (torch.no_grad()):
        for x, t, m in valid_loader:
            x_n_19_81, targets_n_81, mask_n_81 = x.to(device), t.to(device), m.to(device)

            n = x_n_19_81.shape[0]

            logits_n_81_9 = neural_network(x_n_19_81)

            allowed_mask_n_9_81 = x_n_19_81[:, -9:, :]
            allowed_mask_n_81_9 = allowed_mask_n_9_81.transpose(1, 2)

            most_certain_coord, most_certain_indexed_value = get_most_certain_coord_and_index_value(
                allowed_mask_n_81_9=allowed_mask_n_81_9,
                logits_n_81_9=logits_n_81_9,
                target_mask_n_81=mask_n_81
            )

            actual_target_n = targets_n_81[torch.arange(0, n), [c.entry_idx for c in most_certain_coord]]

            diff = torch.isclose(
                input=actual_target_n,
                other=torch.tensor(most_certain_indexed_value).to(torch.float32).to(device),
                rtol=1e-5
            ).float()

            success_rate += diff.sum()

            average_batch_loss = masked_ce_loss(
                logits_n_81_9=logits_n_81_9,
                targets_n_81=targets_n_81,
                mask_n_81=mask_n_81
            )

            total_average_batch_loss += average_batch_loss.item()

    total_average_batch_loss = total_average_batch_loss / len(valid_loader)
    success_rate = success_rate / len(valid_loader.dataset)

    log_info(
        label=f"validation loss:",
        info=f"{total_average_batch_loss:.4f}",
        newline=False
    )

    log_info(
        label=f"success rate with most-certain-algorithm:",
        info=f"{success_rate:.4f}",
        newline=False
    )

    return total_average_batch_loss, success_rate.item()

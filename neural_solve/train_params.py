import torch
from neural_solve.model import SudokuNet

train_data_path: str = "./data/neural-solver-train-data.csv"

device = torch.device("mps")

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


def get_model_weight_path(epoch: int) -> str:
    return f"./data/sudoku_net_training_checkpoints/{epoch}.pth"


sudoku_net = SudokuNet(
    feature_maps=456,
    row_col_box_out_features=27,
    dense_out_features=729
)

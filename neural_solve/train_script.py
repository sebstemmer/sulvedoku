import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from neural_solve.model import SudokuNet
from neural_solve.dataset import SudokuDataset
from common.log_info import log_info
from neural_solve.solve import get_most_certain_coord_and_index_value

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


neural_network = SudokuNet(
    feature_maps=feature_maps,
    row_col_box_out_features=row_col_box_out_features,
    dense_out_features=dense_out_features
).to(device)

if load_from_epoch is not None:
    checkpoint = torch.load(get_model_weight_path(load_from_epoch))

    neural_network.load_state_dict(checkpoint[model_state_key])

    log_info(
        label=f"loaded weights from epoch {load_from_epoch}"
    )

log_info(
    label="SudokuNet:",
    info=str(neural_network)
)

log_info(
    label="number of trainable parameters:",
    info=str(sum(p.numel() for p in neural_network.parameters() if p.requires_grad))
)

log_info(
    label="create test data set..."
)

test_data_set = SudokuDataset(
    data_file_path=train_data_path,
    from_sample_idx_incl=0,
    to_sample_idx_excl=10000
)
test_loader = DataLoader(
    dataset=test_data_set,
    batch_size=test_data_set.__len__(),
    shuffle=False,
)

log_info(
    label="create validation data set..."
)

valid_data_set = SudokuDataset(
    data_file_path=train_data_path,
    from_sample_idx_incl=10000,
    to_sample_idx_excl=40000
)
valid_loader = DataLoader(
    dataset=valid_data_set,
    batch_size=valid_data_set.__len__(),
    shuffle=False,
)

log_info(
    label="create training data set..."
)

train_data_set = SudokuDataset(
    data_file_path=train_data_path,
    from_sample_idx_incl=40000,
    to_sample_idx_excl=int(1e9)
)

train_loader = DataLoader(
    dataset=train_data_set,
    batch_size=batch_size,
    shuffle=True,
)

log_info(
    label="number of training samples:",
    info=str(train_data_set.__len__())
)

optimizer = torch.optim.AdamW(
    params=neural_network.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

if load_from_epoch is not None:
    checkpoint = torch.load(get_model_weight_path(load_from_epoch))

    optimizer.load_state_dict(checkpoint[optimizer_state_key])

    log_info(
        label=f"loaded weights from epoch {load_from_epoch}"
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


def train_one_epoch(loader: DataLoader) -> tuple[float, SudokuNet]:
    neural_network.train()

    epoch_loss: float = 0.0

    for x, target, mask in loader:
        x_n_19_81, targets_n_81, mask_n_81 = x.to(device), target.to(device), mask.to(device)

        optimizer.zero_grad()

        logits_n_81_9 = neural_network(x_n_19_81)

        loss = masked_ce_loss(
            logits_n_81_9=logits_n_81_9,
            targets_n_81=targets_n_81,
            mask_n_81=mask_n_81
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return (epoch_loss / len(loader), neural_network)


def validate(
        neural_network: SudokuNet,
        valid_loader: DataLoader
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

    return total_average_batch_loss, success_rate


log_info(
    label="at start:"
)
valid_loss, valid_success_rate = validate(neural_network, valid_loader)

for epoch in range(num_epochs):
    log_info(label=f"epoch: {epoch}", newline=False)

    train_loss, neural_network = train_one_epoch(train_loader)

    log_info(
        label=f"training loss:",
        info=f"{train_loss:.4f}",
        newline=False
    )

    valid_loss, valid_success_rate = validate(
        neural_network=neural_network,
        valid_loader=valid_loader
    )

    torch.save(
        obj={
            epoch_key: epoch,
            model_state_key: neural_network.state_dict(),
            optimizer_state_key: optimizer.state_dict(),
            train_loss_key: train_loss,
            valid_loss_key: valid_loss,
            valid_success_rate_key: valid_success_rate
        },
        f=get_model_weight_path(epoch)
    )

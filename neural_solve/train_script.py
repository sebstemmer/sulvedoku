import torch
from torch.utils.data import DataLoader

from common.log_info import log_info
from neural_solve.dataset import SudokuDataset
from neural_solve.train_utils import train_data_path, batch_size, learning_rate, weight_decay, load_from_epoch, \
    epoch_key, \
    model_state_key, optimizer_state_key, train_loss_key, valid_loss_key, valid_success_rate_key, num_epochs, \
    get_model_weight_path, SudokuNet, masked_ce_loss, validate, create_sudoku_net_model, training_device

sudoku_net = create_sudoku_net_model().to(training_device)

if load_from_epoch is not None:
    checkpoint = torch.load(get_model_weight_path(load_from_epoch))

    sudoku_net.load_state_dict(checkpoint[model_state_key])

    log_info(
        label=f"loaded weights from epoch {load_from_epoch}"
    )

log_info(
    label="SudokuNet:",
    info=str(sudoku_net)
)

log_info(
    label="number of trainable parameters:",
    info=str(sum(p.numel() for p in sudoku_net.parameters() if p.requires_grad))
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
    params=sudoku_net.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

if load_from_epoch is not None:
    checkpoint = torch.load(get_model_weight_path(load_from_epoch))

    optimizer.load_state_dict(checkpoint[optimizer_state_key])

    log_info(
        label=f"loaded weights from epoch {load_from_epoch}"
    )


def train_one_epoch() -> tuple[float, SudokuNet]:
    sudoku_net.train()

    epoch_loss: float = 0.0

    for x, target, mask in train_loader:
        x_n_19_81, targets_n_81, mask_n_81 = x.to(training_device), target.to(training_device), mask.to(training_device)

        optimizer.zero_grad()

        logits_n_81_9 = sudoku_net(x_n_19_81)

        loss = masked_ce_loss(
            logits_n_81_9=logits_n_81_9,
            targets_n_81=targets_n_81,
            mask_n_81=mask_n_81
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return (epoch_loss / len(train_loader), sudoku_net)


log_info(
    label="at start:"
)
valid_loss, valid_success_rate = validate(
    neural_network=sudoku_net,
    valid_loader=valid_loader,
    device=training_device
)

for epoch in range(num_epochs):
    log_info(label=f"epoch: {epoch}", newline=False)

    train_loss, neural_network = train_one_epoch()

    log_info(
        label=f"training loss:",
        info=f"{train_loss:.4f}",
        newline=False
    )

    valid_loss, valid_success_rate = validate(
        neural_network=neural_network,
        valid_loader=valid_loader,
        device=training_device
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

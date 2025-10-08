import torch
from torch.utils.data import DataLoader

from neural_solve.dataset import SudokuDataset
from neural_solve.train_utils import train_data_path, validate, get_model_weight_path, model_state_key, \
    create_sudoku_net_model

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

epoch: int = 88

sudoku_net = create_sudoku_net_model().to(torch.device("mps"))

checkpoint = torch.load(get_model_weight_path(epoch))
sudoku_net.load_state_dict(checkpoint[model_state_key])

total_average_batch_loss, success_rate = validate(
    neural_network=sudoku_net,
    valid_loader=test_loader,
    device=torch.device("mps")
)
import matplotlib.pyplot as plt
import torch
from neural_solve.train_utils import get_model_weight_path, epoch_key, train_loss_key, valid_loss_key, valid_success_rate_key

epochs: list[int] = []
train_losses: list[float] = []
valid_losses: list[float] = []
valid_success_rates: list[float] = []

for i in range(100):
    data = torch.load(get_model_weight_path(i))
    epochs.append(data[epoch_key])
    train_losses.append(data[train_loss_key])
    valid_losses.append(data[valid_loss_key])
    valid_success_rates.append(data[valid_success_rate_key].item())

plt.figure()
plt.plot(epochs, train_losses, 'x', label="training loss")
plt.plot(epochs, valid_losses, 'x', label="validation loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("cross entropy loss")
plt.show()

plt.figure()
plt.plot(epochs, valid_success_rates, 'x', label="validation success rate")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("success rate")
plt.show()
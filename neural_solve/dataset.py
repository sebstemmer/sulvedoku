import csv

import torch
from torch.utils.data import Dataset

from common.track_progress import track


def get_data(
        data_file_path: str,
        from_sample_idx_incl: int,
        to_sample_idx_excl: int,
) -> list[list[str]]:
    data: list[list[str]] = []
    with open(data_file_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        next(reader)

        for (csv_row_idx, csv_row) in enumerate(reader):
            if csv_row_idx >= to_sample_idx_excl:
                return data

            if csv_row_idx >= from_sample_idx_incl:
                data.append(csv_row)

        return data


class SudokuDataset(Dataset):
    def __init__(
            self,
            data_file_path,
            from_sample_idx_incl,
            to_sample_idx_excl
    ):
        data: list[list[str]] = get_data(
            data_file_path=data_file_path,
            from_sample_idx_incl=from_sample_idx_incl,
            to_sample_idx_excl=to_sample_idx_excl
        )

        self.n = len(data)
        self.x_n_19_81 = torch.zeros((self.n, 19, 81), dtype=torch.float32)
        self.target_n_81 = torch.zeros((self.n, 81), dtype=torch.float32)
        self.mask_n_81 = torch.zeros((self.n, 81), dtype=torch.float32)

        for (sample_idx, sample) in enumerate(data):
            track(
                idx=sample_idx,
                total=self.n,
                output_every=len(data) // 10
            )

            x_sample_1539 = torch.tensor([int(c) for c in sample[0]], dtype=torch.float32)
            x_sample_81_19 = x_sample_1539.reshape(81, 19)
            x_sample_19_81 = x_sample_81_19.permute(1, 0)
            self.x_n_19_81[sample_idx, :, :] = x_sample_19_81

            target_sample_81 = torch.tensor([int(c) for c in sample[1]], dtype=torch.float32)
            self.target_n_81[sample_idx, :] = target_sample_81

            mask_sample_81 = torch.tensor([int(c) for c in sample[2]], dtype=torch.float32)
            self.mask_n_81[sample_idx, :] = mask_sample_81

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (
            self.x_n_19_81[idx],
            self.target_n_81[idx],
            self.mask_n_81[idx]
        )

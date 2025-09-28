import torch
import torch.nn as nn

class RowColBox(nn.Module):
    def __init__(
            self,
            feature_maps,
            out_features
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=19,
                out_channels=feature_maps,
                kernel_size=9,
                stride=9,
                padding=0,
                bias=True
            ),
            nn.ReLU(),
        )

        self.dense = nn.Sequential(
            nn.Linear(
                in_features=feature_maps,
                out_features=out_features,
            ),
            nn.ReLU()
        )

    def forward(self, x):
        n = x.shape[0]

        conv_n_fm_9 = self.conv(x)
        conv_n_9_fm = conv_n_fm_9.transpose(1, 2)
        dense_n_9_out = self.dense(conv_n_9_fm)
        dense_n_9xout = dense_n_9_out.reshape(n, -1)
        return dense_n_9xout


def to_col_format(x):
    n, c, _ = x.shape
    two_d_format = x.reshape(n, c, 9, 9)
    transposed_rows_and_cols = two_d_format.transpose(2, 3)
    return transposed_rows_and_cols.reshape(n, c, 81)


def to_box_format(x):
    n, c, _ = x.shape
    two_d_format = x.reshape(n, c, 9, 9)
    row_row_col_col_four_d = two_d_format.reshape(n, c, 3, 3, 3, 3)
    row_col_row_col_four_d = row_row_col_col_four_d.transpose(3, 4)
    row_col_row_col_two_d = row_col_row_col_four_d.reshape(n, c, 9, 9)
    return row_col_row_col_two_d.reshape(n, c, 81)


class SudokuNet(nn.Module):
    def __init__(
            self,
            feature_maps,
            row_col_box_out_features,
            dense_out_features
    ):
        super().__init__()
        self.row_col_box = RowColBox(
            feature_maps=feature_maps,
            out_features=row_col_box_out_features
        )

        self.dense = nn.Sequential(
            nn.Linear(
                in_features=27 * row_col_box_out_features,
                out_features=dense_out_features
            ),
            nn.ReLU()
        )

        self.out = nn.Linear(
            in_features=dense_out_features,
            out_features=729
        )

    def forward(self, x):
        n, c, _ = x.shape

        row_n_9xrcbout = self.row_col_box(x)

        x_col_n_c_81 = to_col_format(x)
        col_n_9xrcbout = self.row_col_box(x_col_n_c_81)

        x_box_n_c_81 = to_box_format(x)
        box_n_9xrcbout = self.row_col_box(x_box_n_c_81)

        row_col_box_n_27xrcbout = torch.cat([row_n_9xrcbout, col_n_9xrcbout, box_n_9xrcbout], dim=1)

        dense_n_dout = self.dense(row_col_box_n_27xrcbout)

        out_n_729 = self.out(dense_n_dout)

        out_n_81_9 = out_n_729.reshape(n, 81, 9)

        return out_n_81_9

from neural_solve.model import to_box_format
import torch


def to_box_format_test() -> None:
    x = torch.arange(0, 81).unsqueeze(0).unsqueeze(0)

    x_in_box_format = to_box_format(x).squeeze(0).squeeze(0)

    assert torch.equal(x_in_box_format, torch.tensor([
        0, 1, 2, 9, 10, 11, 18, 19, 20, 3, 4, 5, 12, 13, 14, 21, 22, 23, 6, 7, 8, 15, 16, 17, 24, 25, 26, 27, 28, 29,
        36, 37, 38, 45, 46, 47, 30, 31, 32, 39, 40, 41, 48, 49, 50, 33, 34, 35, 42, 43, 44, 51, 52, 53, 54, 55, 56, 63,
        64, 65, 72, 73, 74, 57, 58, 59, 66, 67, 68, 75, 76, 77, 60, 61, 62, 69, 70, 71, 78, 79, 80
    ])) == True


to_box_format_test()

print("all tests passed")

import torch
from torch import Tensor


def map_0_1(input_parameter: Tensor) -> Tensor:
    # return input_parameter / torch.max(input_parameter)
    return torch.clamp(input_parameter,0,1)


def map_sigma_0_1(input_parameter: Tensor) -> Tensor:
    input_parameter = torch.clamp(input_parameter,0,1)
    return input_parameter / torch.sum(input_parameter)

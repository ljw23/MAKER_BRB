import torch
from torch import Tensor

def map_0_1(input_parameter:Tensor )->Tensor:
    return input_parameter/torch.max(input_parameter)


def map_sigma_0_1(input_parameter:Tensor)->Tensor:
    return input_parameter/torch.sum(input_parameter)


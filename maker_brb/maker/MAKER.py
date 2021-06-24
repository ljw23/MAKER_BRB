import torch.nn as nn
from torch.nn import Linear
from torch.nn.parameter import Parameter, UninitializedParameter
from torch import init
from typing import Dict, List
from torch import Tensor


class MAKER(nn.Module):
    def __init__(self,
                 p_h_v_q: Tensor,
                 dim_h: int,
                 dim_v: int,
                 dim_q: int,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MAKER, self).__init__()
        self.p_h_v_q = p_h_v_q
        self.W_m = Parameter(torch.empty(dim_h, dim_v, dim, q))
        self.W_r_hvq = Parameter(torch.empty(dim_h, dim_v, dim, q))
        self.gama = Parameter(torch.empty(dim_v, dim_q, dim_v, dim_q))

    def reset_parameters(self) -> None:
        init.uniform(self.W_m, 0, 1)
        init.uniform(self.W_r_hvq, 0, 1)
        init.uniform(self.W_r_vq, 0, 1)
        init.uniform(self.gama, 0, 1)

    def forward(self, input: Tensor):
        '''
        input.shape: (batchsize, dim_v, dim_q)
        '''
        # m_h_v_q =
        pass

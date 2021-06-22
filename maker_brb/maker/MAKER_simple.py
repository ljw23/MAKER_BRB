import torch.nn as nn
from torch.nn import Linear
from torch.nn.parameter import Parameter, UninitializedParameter
from torch import init
from typing import  Dict,List,Set,Iterable
from torch import Tensor

class MAKER(nn.Module):
    def __init__(self, basic_probility:Dict, index_dict:Dict,hypothesis_set:Set, device=None, dtype=None)->None:
        '''
        {'V1': ['A1', 'B1', 'C1', 'D1'],
        'V2': ['A2', 'B2', 'C2', 'D2'],
        'V3': ['A3', 'B3']}
        '''
        factory_kwargs = {'device':device, 'dtype':dtype}
        super(MAKER, self).__init__()

        self.basic_probility = basic_probility

        self.w_m_hvq = {}
        self.w_r_hvq = {}
        self.index_set = []
        for h in hypothesis_set:
            for v_q in index_dict.items():
                v, q_list = v_q
                for q in q_list:
                    
                    self.build_hvq()
    
    def build_hvq(self, indexname:str):
        if set(indexname) not in self.index_set_list:
            self.w_m_hvq[indexname] = Parameter(torch.Tensor(1))
            self.w_r_hvq[indexname] = Parameter(torch.Tensor(1))
        for _index_set in index_set:
            if set(indexname).update(_index_set) not in index_set_list:
                new_index_set = set(indexname).update(_index_set)
                multi_index_name = self.get_feature_name(new_index_set)
                self.w_m_hvq[multi_index_name] = Parameter(torch.Tensor(1))
                self.w_r_hvq[multi_index_name] = Parameter(torch.Tensor(1))
                self.index_set_list.append(new_index_set)


            


        
                





    def reset_parameters(self) -> None:
        init.uniform(self.W_m, 0 ,1 )
        init.uniform(self.W_r_hvq, 0 ,1 )
        init.uniform(self.W_r_vq, 0 ,1 )
        init.uniform(self.gama_v1q1_v2q2, 0 ,1 )

    def forward(self, input: Tensor):
        '''
        input.shape: (dim_v, dim_q)
        '''        
        m_h_v_q = 


from pydantic import BaseModel
from typing import Dict, List,Set
from pprint import pprint
import torch
from .attrbute import Attribute_info

class Encoded_Parameter:
    def __init__(self,parameter_id :int=-1,
                attribute_combination: List[Attribute_info] = None,
                tensor: torch.Tensor = torch.tensor(1.0)):
        self.parameter_id = parameter_id #参数张量的id
        self.attribute_combination = attribute_combination
        self.tensor = tensor

class Encoded_Parameters:
    def __init(self,parameter_list: List[Encoded_Parameter]=None):
        super(Encoded_Parameters, self).__init__()
        self.parameter_list = parameter_list or []

    def get(self,**kwargs)->torch.Tensor:
        raise NotImplementedError

    def from_id_get_tensor(self,parameter_id :int)->torch.Tensor:
        for parameter in self.parameter_list:
            if parameter_id == parameter.parameter_id:
                return parameter.tensor

    def from_attribute_get_tensor(self,attribute_combination:List[Attribute_info]):
        






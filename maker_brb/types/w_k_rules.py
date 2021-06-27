import torch
from .encoded_parameter import Encoded_Parameters, Encoded_Parameter
from .attrbute import build_combined_atrributes


class W_k_rules(Encoded_Parameters):
    def __init__(self):
        super(W_k_rules, self).__init__()
        attribute_list = build_combined_atrributes()
        self.build_parameter_list(attribute_list)

    def build_parameter_list(self, attribute_list):
        self.parameter_list = []
        for i, attribute in enumerate(attribute_list):
            self.parameter_list.append(
                Encoded_Parameter(parameter_id=i,
                                  attribute_combination=attribute,
                                  tensor=torch.tensor(1.0)))

    def get(self, k: int):
        return self.from_id_get_tensor(parameter_id=k)

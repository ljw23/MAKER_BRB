import torch
from torch.nn import Parameter
from torch import  Tensor
from .rule_belief_layer import Rule_belief_layer
from .rule_weight_layer import Rule_weight_layer
from maker_brb.utils.mapping_func import *

class BRB_Model(torch.nn.Module):
    def __init__(self, num_k:int, num_A:int, num_h:int):
        '''
        num_k: 规则数量
        num_A: 特征数量
        num_h: 假设数量
        '''
        super(BRB_Model, self).__init__()
        
        self.num_h =num_h
        self.num_A = num_A
        self.num_k = num_k

        self.rule_weight_layer = Rule_weight_layer(num_k, num_A)
        self.rule_belief_layer =Rule_belief_layer(num_k, num_h)
        
    def forward(self, input_x:Tensor):
        '''
        input_x: [batch_size, num_A] 一个batch的输入数据，每个数据有num_A个特征的维度
        '''
        W_rule_act = self.rule_weight_layer(input_x) #[batch_szie, num_k]
        p_n = self.rule_belief_layer(W_rule_act) # [batch_size, num_h]
        return p_n

    @torch.no_grad()
    def post_mapping(self):
        self.rule_weight_layer.theta = Parameter(map_0_1(self.rule_weight_layer.theta))

        for k in range(self.num_k):
            self.rule_weight_layer.delta_[k] = Parameter(map_0_1(self.rule_weight_layer.delta_[k]))
            self.rule_belief_layer.beta[k] = Parameter(map_sigma_0_1(self.rule_belief_layer.beta[k]))

    






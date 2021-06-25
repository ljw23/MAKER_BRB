import torch
from torch.nn import Parameter
from torch import  Tensor
from .rule_belief_layer import Rule_belief_layer
from .rule_weight_layer import Rule_weight_layer

class BRB_Model(torch.nn.Module):
    def __init__(self, num_k:int, num_A:int, num_h:int):
        '''
        num_k: 规则数量
        num_A: 特征数量
        num_h: 假设数量
        '''
        super(BRB_Model, self).__init__()
        
        self.num_h =num_h
        self.data_transformer = Data_tranformer() #将input_x转换为alpha
        self.num_A = self.data_transformer.num_A
        self.num_k = self.data_transformer.num_k
        self.rule_weight_layer = Rule_weight_layer(num_k, num_A)
        self.rule_belief_layer =Rule_belief_layer(num_kalpha, num_h)
        
    def forward(self, input_x:Tensor):
        '''
        input_x: [batch_size, num_A] 一个batch的输入数据，每个数据有num_A个特征的维度
        '''
        alpha = self.data_transformer(input_x) #[batch_size,num_k,num_A]
        W_rule_act = self.rule_weight_layer(alpha)
        p_n = self.rule_belief_layer(W_rule_act)
        return p_n





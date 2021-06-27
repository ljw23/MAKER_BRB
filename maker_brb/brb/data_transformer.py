import torch
from torch.nn.functional import softmax
from torch.nn import Parameter
from torch import  Tensor
from torch.nn import init
from typing import Dict

class  Data_transformer(torch.nn.Module):
    '''
    由于采用投影法对约束问题优化，强制条件sum(beta)=1.
    对论文中公式进行了化简
    '''
    def __init__(self, attribute_dict:Dict) -> None:
        '''
        attribute_dict:Dict 特征属性的字典 {A_i: A_v,i }
        {A1:[0,1,2], A2:[1.1, 1,5, 3.0, 4.0]}
        input_x 与特征属性字典格式一致
        '''
        super(Data_transformer, self).__init__()
        
        self.num_A = len(attribute_dict)
        ##需要有2个字典， input_x 的i 项对应的特征
        ##第k条规则对应特征[k,num_A]数组

    
    def forward(self, input_x:Tensor) -> alpha:Tensor:
        '''
        input_x: [batch_size, num_A_v] 一个batch的输入数据，每个数据有num_A个特征的维度

        alpha: [batch_size,num_k,num_A] 每一条规则在每个特征下
        '''
        

            








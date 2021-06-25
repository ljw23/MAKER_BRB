import torch
from torch.nn.functional import  softmax
from torch.nn import Parameter
from torch import  Tensor

class Rule_weight_layer(torch.nn.Module):
    def __init__(self, num_k:int, num_A:int ):
        '''
        delta_: [k,v] 每个规则对于每个特征的权重，可人为设置，并不参与训练
        '''
        super(Rule_weight_layer, self).__init__()
        self.num_A = num_A
        self.num_k = num_k
        self.theta = Parameter(torch.ones(num_k)) 
        self.delta_ = Parameter(torch.ones([num_k, num_A]))
        

    def forward(self, alpha:Tensor):
        '''
        alpha: 对于一条数据x_m，各特征的匹配度，可设为二维[v,q]。
               由于对于1条规则而言，各特征只会取一类，因此维度转换为[k,v]
        '''
        W_rule_act = torch.Tensor(self.num_k)

        for k in range(self.num_k):
            product_alpha = 1
            for i in range(self.num_A):
                product_alpha *= torch.pow(alpha[k,i], self.delta_[k,i])
            W_rule_act[k] = self.theta[k]*product_alpha

        W_rule_act = W_rule_act/torch.sum(W_rule_act)
        return W_rule_act






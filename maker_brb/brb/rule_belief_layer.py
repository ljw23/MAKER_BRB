import torch
from torch.nn.functional import softmax
from torch.nn import Parameter
from torch import Tensor
from torch.nn import init
import math


class Rule_belief_layer(torch.nn.Module):
    '''
    由于采用投影法对约束问题优化，强制条件sum(beta)=1.
    对论文中公式进行了化简
    '''
    def __init__(self, num_k: int, num_h: int) -> None:
        '''
        beta: [k,n]  k为规则数,n为假设数，本layer只有此参数需要优化，也可以由专家制定，不优化
        num_k:规则数
        num_h: 假设数
        '''
        super(Rule_belief_layer, self).__init__()
        self.num_h = num_h
        self.num_k = num_k
        self.beta = Parameter(torch.empty(self.num_k, self.num_h))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        init.kaiming_uniform_(self.beta,
                              a=math.sqrt(5))  #belief degree随机初始化，可专家规则制定。
        self.beta.data = softmax(self.beta, dim=1)

    def forward(self, W_rule_act: Tensor):
        '''
        对论文公式进行了化简，采用p/sum(p)公式代替分母
        W_rule_act: [batchsize, num_k, ] 规则激活权重 
        '''
        batch_size = W_rule_act.shape[0]
        p_n = torch.empty(batch_size, self.num_h)
        for n in range(self.num_h):
            product_term1 = torch.ones(batch_size)  # W*beta+1-W
            product_term2 = torch.ones(batch_size)
            for k in range(self.num_k):
                product_term1 *= W_rule_act[:, k] * self.beta[
                    k, n] + 1 - W_rule_act[:, k]
                product_term2 *= 1 - W_rule_act[:, k]

            p_n[:, n] = product_term1 - product_term2

        p_n = torch.div(p_n.T, torch.sum(p_n, dim=-1)).T
        return p_n

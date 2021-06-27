# 定义一个 my_layer.py
import torch
from torch.nn import Parameter

# class MyLayer(torch.nn.Module):
#     '''
#     因为这个层实现的功能是：y=weights*sqrt(x2+bias),所以有两个参数：
#     权值矩阵weights
#     偏置矩阵bias
#     输入 x 的维度是（in_features,)
#     输出 y 的维度是（out_features,) 故而
#     bias 的维度是（in_fearures,)，注意这里为什么是in_features,而不是out_features，注意体会这里和Linear层的区别所在
#     weights 的维度是（in_features, out_features）注意这里为什么是（in_features, out_features）,而不是（out_features, in_features），注意体会这里和Linear层的区别所在
#     '''
#     def __init__(self):
#         super(MyLayer, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数

#         self.weight = torch.nn.Parameter(torch.Tensor(1)) # 由于weights是可以训练的，所以使用Parameter来定义

#     def forward(self, input):
#         input_=torch.pow(input,2)
#         y=torch.mul(input_,self.weight)
#         return y


class MyLayer(torch.nn.Module):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.alpha1 = Parameter(torch.Tensor(1))
        # self.delta1 = Parameter(torch.empty(1))
        # self.alpha2 = Parameter(torch.empty(1))
        # self.delta2 = Parameter(torch.empty(1))

        # self.reset_parameters()

        # self.weight1 = torch.tensor(0.6,requires_grad=True)
        # self.weight2 = torch.tensor(2.0,requires_grad=True)

        # self.weight1 = Parameter(torch.tensor(1.0))
        # self.weight2 = Parameter(torch.tensor(2.0))
    @torch.no_grad()
    def reset_parameters(self) -> None:
        # self.weight1 = torch.tensor(1.0, requires_grad=True)
        # self.weight2 = torch.tensor(1.0, requires_grad=True)
        # constant(self.weight1, 0.6)
        # constant(self.weight2, 2.0)
        # torch.nn.init.ones_(self.weight1)
        # torch.nn.init.ones_(self.weight2)
        init.uniform_(self.alpha1, 0, 1)
        # init.uniform_(self.delta1, 0, 1)
        # init.uniform_(self.alpha2, 0, 1)
        # init.uniform_(self.delta2, 0, 1)

    def forward(self, x):
        input_ = torch.pow(x, 2)
        y = torch.mul(input_, self.alpha1)
        return y

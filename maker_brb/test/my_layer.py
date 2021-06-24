# 定义一个 my_layer.py
import torch
 
class MyLayer(torch.nn.Module):
    '''
    因为这个层实现的功能是：y=weights*sqrt(x2+bias),所以有两个参数：
    权值矩阵weights
    偏置矩阵bias
    输入 x 的维度是（in_features,)
    输出 y 的维度是（out_features,) 故而
    bias 的维度是（in_fearures,)，注意这里为什么是in_features,而不是out_features，注意体会这里和Linear层的区别所在
    weights 的维度是（in_features, out_features）注意这里为什么是（in_features, out_features）,而不是（out_features, in_features），注意体会这里和Linear层的区别所在
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(MyLayer, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features)) # 由于weights是可以训练的，所以使用Parameter来定义
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(in_features))             # 由于bias是可以训练的，所以使用Parameter来定义
        else:
            self.register_parameter('bias', None)
 
    def forward(self, input):
        input_=torch.pow(input,2)+self.bias
        y=torch.matmul(input_,self.weight)
        return y
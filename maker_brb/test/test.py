import torch.nn as nn
import torch
from torch.nn import Parameter
from torch.nn.init import constant
from torch.nn import init
from torch.nn.modules import Linear
# from torch.functional import 

class MyLayer(nn.Module):
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
        input_=torch.pow(x,2)
        y=torch.mul(input_,self.alpha1)     
        return y

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()  # 第一句话，调用父类的构造函数
        self.mylayer1 = MyLayer()
    
    def forward(self, x):
        x = self.mylayer1(x)
        return x

if __name__ == '__main__':
    model = MyNet()
    print(model.parameters())
    learning_rate = 0.01
    input_x = torch.tensor(1)
    input_y = torch.tensor(5)
    
    # len_input = len(input_x)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(10):
        output = model(input_x)
        loss = criterion(torch.Tensor([[output]]), torch.Tensor([[input_y]]))
        loss.requires_grad = True
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss:%f'%loss)



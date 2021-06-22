from torch.nn import Module
from torch.nn import ReLU,Linear

class Pow_Model(Module):
    def __init__(self, x, y):
        super(Module, self).__init__()
        self.func = torch.pow()
    
    def forward(self,x):
        return torch.pow(y,x)
        


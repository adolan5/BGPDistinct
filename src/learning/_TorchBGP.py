import torch
from torch import nn

class DistinctNN(nn.Module):
    """Main neural network definition for BGPDistinct.
    Used to perform simple classification of BGP data to determine which
    messages are distinct and which are simply propagations.
    """
    def __init__(self):
        super(DistinctNN, self).__init__()
        # Naive structure first; input > h1 > h2 > out
        self.h1 = nn.Linear(4, 100)
        self.a1 = nn.ReLU()
        self.h2 = nn.Linear(100, 100)
        self.a2 = nn.ReLU()
        self.raw_out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        out = self.h1(x)
        out = self.a1(out)
        # out = self.h2(out)
        # out = self.a2(out)
        out = self.raw_out(out)
        return out

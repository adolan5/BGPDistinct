import torch
from torch import nn

class DistinctNN(nn.Module):
    """Main neural network definition for BGPDistinct.
    Used to perform simple classification of BGP data to determine which
    messages are distinct and which are simply propagations.
    """
    def __init__(self):
        super().__init__()
        # Naive structure first; input > h1 > h2 > out
        self.h1 = nn.Linear(4, 4)
        self.relu1 = nn.ReLU()
        self.h2 = nn.Linear(4, 4)
        self.relu2 = nn.ReLU()
        self.raw_out = nn.Linear(4, 2)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        h1_a = self.relu1(self.h1(x))
        h2_a = self.relu2(self.h2(h1_a))
        out = self.out_act(self.raw_out(h2_a))
        return out

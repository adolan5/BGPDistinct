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
        self.h1 = nn.Linear(6, 100)
        self.relu1 = nn.ReLU()
        self.h2 = nn.Linear(100, 100)
        self.relu2 = nn.ReLU()
        self.raw_out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

        self.hidden = nn.Sequential(self.h1, self.relu1,
                self.h2, self.relu2,
                self.raw_out)

    def forward(self, x):
        out = self.hidden(x)
        out = self.out_act(out)
        print(out)
        return out

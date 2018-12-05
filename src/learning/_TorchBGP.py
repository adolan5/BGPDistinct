import torch
from torch import nn

class DistinctNN(nn.Module):
    """Main neural network definition for BGPDistinct.
    Used to perform simple classification of BGP data to determine which
    messages are distinct and which are simply propagations.
    """
    def __init__(self, n_hidden):
        super(DistinctNN, self).__init__()
        # Naive structure first; input > h1 > h2 > out
        self.h1 = nn.Linear(4, n_hidden)
        self.a1 = nn.ReLU()
        self.h2 = nn.Linear(n_hidden, n_hidden)
        self.a2 = nn.ReLU()
        self.h3 = nn.Linear(n_hidden, n_hidden)
        self.a3 = nn.ReLU()
        self.raw_out = nn.Linear(n_hidden, 2)
        self.out_act = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.h1(x)
        out = self.a1(out)
        out = self.h2(out)
        out = self.a2(out)
        out = self.h3(out)
        out = self.a3(out)
        out = self.raw_out(out)
        out = self.out_act(out)
        return out

    def single_iteration(self, X, T, optimizer, loss_func):
        """Perform a single iteration of training of the network.
        Args:
        X: The inputs.
        T: The target values.
        optimizer: The torch.optim optimizer to use.
        loss_func: The torch.nn loss function to use.
        Returns:
        The accumulated errors for this iteration of learning.
        """
        losses = []
        # Forward pass
        out = self(X)
        #Calculate loss
        loss = loss_func(out, T.view(-1))
        # Zero gradients
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # Update weights
        optimizer.step()
        # Track errors
        losses.append(loss.data.numpy())
        return losses

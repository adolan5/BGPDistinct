import torch
from collections import OrderedDict
from torch import nn

class DistinctNN(nn.Module):
    """Main neural network definition for BGPDistinct.
    Used to perform simple classification of BGP data to determine which
    messages are distinct and which are simply propagations.
    """
    def __init__(self, n_hidden=2, n_neurons=20):
        """Constructor.
        Creates the network structure in a dynamic way.
        Args:
        n_hidden: The number of hidden layers to use in this network; that is,
            the number of layers between input and output.
        n_neurons: The number of neurons to have in each hidden layer.
        """
        super(DistinctNN, self).__init__()
        # Naive structure first; input > h1 > h2 > out
        self.h0 = nn.Linear(4, n_neurons)
        self.a0 = nn.ReLU()
        hiddens = OrderedDict()

        # Define a hidden layer sequence
        for i in range(1, n_hidden + 1):
            hiddens['h{}'.format(i)] = nn.Linear(n_neurons, n_neurons)
            hiddens['a{}'.format(i)] = nn.ReLU()
        self.hidden_sequence = nn.Sequential(hiddens)
        self.raw_out = nn.Linear(n_neurons, 2)
        self.out_act = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.h0(x)
        out = self.a0(out)
        out = self.hidden_sequence(out)
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
        losses.append(loss.data)
        return losses

from torch import nn
from collections import OrderedDict

class FirstNetworkStruct(nn.Module):
    """The initial network strucutre used for BGPDistinct.
    Recreated here for inclusion in the final report.
    This network modeled a regression problem, not a true classification.
    """
    def __init__(self, num_neurons=10):
        super(FirstNetworkStruct, self).__init__()
        self.inputs = nn.Linear(4, num_neurons)
        self.act = nn.ReLU()
        hiddens = OrderedDict()
        for i in range(1, 3):
            hiddens['h{}'.format(i)] = nn.Linear(num_neurons, num_neurons)
            hiddens['a{}'.format(i)] = nn.ReLU()
        self.hidden = nn.Sequential(hiddens)
        self.output = nn.Linear(num_neurons, 1)

    def forward(self, x):
        out = self.inputs(x)
        out = self.act(out)
        out = self.hidden(out)
        out = self.output(out)
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
        loss = loss_func(out, T.double())
        # Zero gradients
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # Update weights
        optimizer.step()
        # Track errors
        losses.append(loss.data)
        return losses

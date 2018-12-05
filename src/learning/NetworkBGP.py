import copy
import numpy as np
import random
import torch
from labeling import Labeler
from learning._TorchBGP import DistinctNN
from learning.DataRescaler import DataRescaler

class NetworkBGP:
    """The NetworkBGP class.
    This class is responsible for all of the machine learning aspects of the
    BGPDistinct project.
    This class can be used to create, train, and use neural networks on BGP
    data that has been formatted and had features extracted, for the purposes
    of investigating distinct BGP events.
    """
    def partition(data, train_size=0.8):
        """Static partition method.
        Performs partition by "plucking" (1 - train_size) * n messages out of the
        provided data set, and places them in a new test set.
        Args:
        data (list): The original formatted BGP data.
        test_size (float): The proportion of the data to use as a training set.
        Returns:
        A tuple containing the partitioned training input and target sets as tensors.
        """
        # Start with a copy, will be training
        train = copy.deepcopy(data)
        test = []

        # Get distribution of message indices, keep ordering
        test_len = int((1 - train_size) * len(data))
        test_indices = sorted(random.sample(range(len(data)), test_len))

        # For each index, remove from train and append to test
        for i in test_indices:
            train.remove(data[i])
            test.append(data[i])

        # Now label each set individually (performed in place)
        Labeler(train)
        Labeler(test)

        # Rescale data as well
        train = DataRescaler(train).scaled_data
        test = DataRescaler(test).scaled_data

        # Convert to tensors
        # Inputs
        Xtrain = torch.tensor([[s.get('time')] + list(s.get('composite').values()) for s in train], dtype=torch.double)
        Xtest = torch.tensor([[s.get('time')] + list(s.get('composite').values()) for s in test], dtype=torch.double)

        # Targets
        Ttrain = torch.tensor([[s.get('distinct')] for s in train], dtype=torch.double)
        Ttest = torch.tensor([[s.get('distinct')] for s in test], dtype=torch.double)
        return(Xtrain, Ttrain, Xtest, Ttest)

    def __init__(self, data, n_hidden):
        """Constructor.
        Initializes basic structures.
        Args:
        data (list): The data on which to train.
        n_hidden (int): The number of hidden layer outputs.
        """
        # Original data
        self._data = data

        # First network
        self.net = DistinctNN(n_hidden).double()

        # First partitioned data
        self.Xtrain, self.Ttrain, self.Xtest, self.Ttest = NetworkBGP.partition(self._data)

    def _get_net(self):
        """Get a fresh instance of the BGPDiscint neural net."""
        net = DistinctNN()
        # Set to use 64-bit floating-point
        net.double()
        return net

    def train_network(self, num_iterations=1):
        """Train the network of this instance for a number of iterations."""
        # Create optimizer and loss function; keeping things simple for now
        # optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.1)
        loss = torch.nn.MSELoss()

        # TODO: Perform the actual training
        losses = []
        for i in range(num_iterations):
            # Now run an iteration
            losses.append(self._single_iteration(self.Xtrain, self.Ttrain, optimizer, loss))
            """
            if (i % 10 == 0):
                print(list(self.net.parameters()), end='\r')
            """

        return losses

    def _single_iteration(self, X, T, optimizer, loss_func):
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
        out = self.net(X)

        #Calculate loss
        loss = loss_func(out, T)

        # Zero gradients
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()

        # Update weights
        optimizer.step()

        # Track errors
        losses.append(loss.data.numpy())
        return losses

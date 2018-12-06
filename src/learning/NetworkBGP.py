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
    def partition(data, device, train_size=0.8):
        """Static partition method.
        Performs partition by "plucking" (1 - train_size) * n messages out of the
        provided data set, and places them in a new test set.
        Args:
        data (list): The original formatted BGP data.
        device (str): The device on which the tensors should be created/stored.
        train_size (float): The proportion of the data to use as a training set.
        Returns:
        A tuple containing the partitioned training input and target sets as tensors.
        """
        # Start with a copy, will be training
        train = copy.deepcopy(data)
        test = []

        # Get distribution of message indices, keep ordering
        test_len = int((1 - train_size) * len(data))
        test_indices = sorted(random.sample(range(len(data)), test_len), reverse=True)

        # For each index, remove from train and append to test
        for i in test_indices:
            test.append(train.pop(i))

        # Need to reverse test now
        test.reverse()

        # Now label each set individually (performed in place)
        Labeler(train)
        Labeler(test)

        # Rescale data as well
        train = DataRescaler(train).scaled_data
        test = DataRescaler(test).scaled_data

        # Convert to tensors
        # Inputs
        Xtrain = torch.tensor([[s.get('time')] + list(s.get('composite').values()) for s in train], dtype=torch.double).to(device)
        Xtest = torch.tensor([[s.get('time')] + list(s.get('composite').values()) for s in test], dtype=torch.double).to(device)

        # Targets
        Ttrain = torch.tensor([[s.get('distinct')] for s in train], dtype=torch.long).to(device)
        Ttest = torch.tensor([[s.get('distinct')] for s in test], dtype=torch.long).to(device)
        return(Xtrain, Ttrain, Xtest, Ttest)

    def __init__(self, data, n_hidden, n_neurons, lr=0.01, cw=[0.1, 1], train_rat=0.8, force_cpu=False):
        """Constructor.
        Initializes basic structures.
        Args:
        data (list): The data on which to train.
        n_hidden (int): The number of hidden layer outputs.
        n_neurons (int): The number of neurons per hidden layer.
        lr (float): The learning rate to use for this network.
        cw (list of float): The weights to assign each class for loss calculation.
        train_rat (float): The ratio that the training set should make up of
            the data partition.
        force_cpu (bool): True if only the CPU should be used, even if CUDA is available.
        """
        # Check if CUDA is available and use it if we selected to
        self._device = 'cpu'
        if torch.cuda.is_available() and not force_cpu:
            self._device = 'cuda'
        # Original data
        self._data = data

        # First network
        self.net = DistinctNN(n_hidden, n_neurons).double().to(self._device)

        # Set learning rate
        self.learning_rate = lr

        # Set class weights
        self.class_weights = cw

        # First partitioned data
        self.Xtrain, self.Ttrain, self.Xtest, self.Ttest = NetworkBGP.partition(self._data, self._device, train_rat)

    def train_network(self, num_iterations=1):
        """Train the network of this instance for a number of iterations."""
        # Create optimizer and loss function; keeping things simple for now
        # optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        loss = torch.nn.NLLLoss(weight=torch.tensor(self.class_weights).double().to(self._device))

        # TODO: Perform the actual training
        losses = []
        for i in range(num_iterations):
            # Now run an iteration
            losses.append(self.net.single_iteration(self.Xtrain, self.Ttrain, optimizer, loss))
            """
            if (i % 10 == 0):
                print(list(self.net.parameters()), end='\r')
            """

        return losses

    def get_predicted_classes(self, output):
        """Get the predicted classes from the output of the network.
        Args:
        output (tensor): The output from the network when used on an input.
        Returns:
        A numpy array containing the predicted ouptut classes.
        """
        return torch.max(output, 1)[1].cpu().numpy()

    def get_correct(self, predicted, actual):
        """Get the ratios of correctly predicted classes overall, and for each
        individual class.
        Args:
        predicted (np.ndarray): The array of predicted classes derived from the
            outputs of the network.
        Returns:
        A tuple containing the ratios of correct:incorrect for all data points,
        data points that should be distinct, and data points that should not be
        distinct.
        """
        ret_ratios = [np.sum(predicted == actual) / len(actual)]
        for i in range(2):
            actual_ones = np.where(actual == i)[0]
            should_be_ones = np.take(predicted, actual_ones)
            actual_ones = np.take(actual, actual_ones)
            ret_ratios.append(np.sum(should_be_ones == actual_ones) / len(actual_ones))

        return tuple(ret_ratios)

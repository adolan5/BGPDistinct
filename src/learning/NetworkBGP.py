import copy
import numpy as np
import random
from labeling import Labeler
from learning._TorchBGP import DistinctNN

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
        A tuple containing the training set and the test set.
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

        # Now label each set individually (performed in place) and return
        Labeler(train)
        Labeler(test)
        return(train, test)

    def __init__(self, data):
        """Constructor.
        Initializes basic structures.
        """
        # Original data
        self._data = data

        # Network
        self.net = DistinctNN()

# Neural Network Definitions
*For more detailed information, see the main project [notebook](/notebooks/BGPDistinct.ipynb)*.

## Network Manager: `NetworkBGP`
This class is used to manage different aspects of neural network
experimentation. It is constructed with a formatted and extracted set of BGP
data, as well as a neural network. The main operations performed by this class
are:
* Partitioning of data
* Training of the network
  * This function takes an optimizer and a loss function to be used during
    training.
* A utility function to get ratios of correct classification
  * I.E., the return value of this function is `(overall_correct,
    duplicate_correct, distinct_correct)`

## `FirstNetworkStruct`
This class represents the first network strucutre that was created. This network
modeled a linear regression, not a true classification, and was used mostly as a
proof of concept example.

This network can be constructed with an arbitrary number of neurons for each of
its two hidden layers, and is expected to be used with SGD and MSELoss.

## `SecondNetworkStruct`
This class is the final network structure that took the main focus of this
project. It actually models a classification problem with the addition of the
*LogSoftMax* function for the output layer and the *NLLLoss* function for
computing the loss.

This network can be constructed with both an arbitrary number of hidden layers
and an arbitrary number of neurons per hidden layer.

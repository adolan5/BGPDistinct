# The `learning` Package
*The main machine learning aspects of BGPDistinct.*

**For basic information on the neural net definitions, see the
[Networks](./Networks.md)** document.

## Introduction
This package contains utilities and basic code for defining neural networks to
be used on BGPMon data that has been formatted, extracted, and labeled. The
machine learning aspect boils down to a simple classification problem, though
there are some caveats. The ideal trained network would be able to faithfully
"count" up the number "distinct" BGP messages that occurred during a certain
time frame.

## Partitioning
Partitioning data for a machine learning classification problem is typically
straightforward; however, this BGP data is unique in that the distinctness of
any one message is determined by what messages the precede it.

### The Trouble with Ordering
BGP messages are considered distinct if no message that shares its key
characteristics (prefix, destination, etc.) occurred before it within a certain
time frame. It follows that the data must be kept in time-order when
partitioned. In other words, the data cannot be shuffled in such a way that
would make messages appear out of order with respect to time.

### The Trouble with Labeling
When data is partitioned, it becomes necessary to re-label each partition. As
discussed before, a message is only considered distinct if it meets some
criteria dependent on the messages that came before it. Therefore, data cannot
simply be labeled and then partitioned, for there may be some messages that,
when moved to a different partition, become distinct when they were not before.

### Partitioning Methodology
With these caveats in mind, partitioning will proceed as follows:
* Begin with the set of formatted and extracted BGP data, of *n* messages.
* Choose (0.2 * *n*) random indices, ranging from 0 to *n* - 1.
* Move the messages at those indices into a new set of data, the testing set,
  while preserving their original order with respect to time.
* Label both the training and test partitions separately.

## Scaling and Formatting for Machine Learning
We must represent our data in acceptable ways for use with machine learning
implementaitons. Therefore, another step of preprocessing is required. This step
is kept separate from the above step of extraction in order to preserve
readability for regular data experimentation. This step includes conversions of
prefixes to numbers, and the re-scaling of all data points. The former presents
its own unique challenges, described below. Note that labeling of the data
should take place after it has been preprocessed, as the data is in its most
explicit form at this point.

### The Issues with Prefixes
#### 1: Format
In order to use prefixes as input to a neural network, they should be
represented as numbers. There is a clean conversion for IPv4, which are merely
32 bit addresses. IPv6 is a different story (see below).  It is important to
note, as well, that PyTorch only uses signed 64-bit integers. Why? Who knows-
but this also must be accounted for.

#### 2: IPv6
IPv6 addresses are 128 bits in length, and cannot be represented in 64 bit
numbers, like those of PyTorch and Cuda. The solution to this problem for now is
to exploit the fact that the first 4 octets (64 bits) of an IPv6 address are
used for routing purposes, while the last 4 octets are used as an interface
identifier. However, it is still possible that announcements contain more octets
than they should- we must therefore capture the entire address for any IP. We
can achieve this by splitting any address in half, into two 16 bit integers for
IPv4, or 2 64 bit integers for IPv6.

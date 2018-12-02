# The `learning` Package
*The main machine learning aspects of BGPDistinct.*

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

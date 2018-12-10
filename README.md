# BGPDistinct
*A measure of the number of distinct BGP messages over time.*
##### Author: Andy Dolan

## Navigating and Using this Repository
For demonstrated use of most components of this project, use the main
BGPDistinct [notebook](./notebooks/BGPDistinct.ipynb). This document provides a
background on BGP and is where primary experiments took place; results and
discussion are captured there.

## Introduction
The Border Gateway Protocol (BGP) is a foundational technology of today's
Internet. In essence, this protocol allows different autonomous systems (ASes)
to announce to each other how to reach some prefix (i.e., a set of addresses in
the Internet address space).

This idea takes the similar form of a simple game of [telephone](https://en.wikipedia.org/wiki/Chinese_whispers).
An AS with AS number 54321 that owns some prefix, such as `10.0.0.0/24`,
announces this ownership, telling its immediate neighboring BGP routers "to get
to any address with the prefix `10.0.0.0/24`, you should come to AS 54321." In
turn, the neighbors that receive this message **propagate** it to their
neighbors, appending their own AS number to the path, and thus informing
neighbors of how to get to the prefix, by means of a number of hops to the final
AS, the owner of the prefix.

## Problem Statement
In the game of telephone, a single round consists of a single message being
propagated from player to player. The same concept exists in BGP, wherein a
single announcment of ownership of a prefix will propagate to a potentially
large number of different ASes, each of which may interpret it and pass it along
differently.

Previous work has used the number of captured BGP messages as a metric for
different characterizations (e.g., presence of an attack), but how does the
number of *distinct* events compare?

#### Definition: Distinct Event
For the purposes of this project, a **distinct event** is a BGP
announcement/advertisement of the ownership of a prefix within a certain time
frame. That is, many different ASes may propagate messages related to this
event, prepending themselves to its growing AS\_PATH, but all messages relate to
a single prefix, owned by a single AS, at a particular time. To be clear, in
terms of time, the same BGP event may occur many times at different points in
time (consider the example that a particular AS announces its ownership of a
prefix daily at the same time). This project also aims to determine some
estimate of how temporal differences can define different distinct events.

In not so many words, the issue that this project aims to address is that of
counting the number of these *distinct events* for a given time frame; in other
words, to count how many games of telephone are being played during this
time frame.

## BGP Data
The data for this project will be gathered from the [BGPMon](www.bgpmon.io)
project, which aggregates BGP data from a number of "collectors" around the
world. For more information on the data, see the `data/` [README](./data).

## Methodologies and Challenges
Simple machine learning techniques will be employed for the purposes of this
project. That is, some simple set of data will be labeled and a neural network
of some kind will be trained and tested using said data.

The interesting aspect of this methodology is the fact that this is a
classification problem, but in a dynamic sense. Different BGP messages will be
classified into one of two classes, (1) having been seen before, and (2) having
not been seen before. These classes are determined, of course, by the messages
that have already been processed.

## *TODO*
*The following outline lists what general tasks need to be completed. Those
tasks marked with ~~strikethrough~~ have been completed.*
* ~~Exploration of data, familiarization with BGP4 specification and data
  adherence.~~
* ~~Preprocessing of data, both in validation of format (JSON) and filtering by
  appropriate message type (announcements).~~
  * ~~Refinement of preprocessing, to extract only the features relevant to
    machine learning.~~
* ~~Investigation into how to best distinguish different messages as pertaining to
  one event or another.~~
* ~~Manual labeling of some subset of data.~~
  * ~~Programmatically, or truly manually?:~~
  * ~~Programmatically, creation of a routine that allows a user to label messages
    that may or may not be "distinct."~~
* ~~Implement simple partitioning of data.~~
* ~~Introduction of PyTorch libraries and initial training and analysis.~~
* ~~Tuning of machine learning for optimal results.~~

## Future Work
*The following items should be moved to the TODO section if they are taken on.*
* Revise the labeling routine to account for temporal data.
* Consider temporal data during classification (this goes hand in hand with
  revising the labeling scheme).
* Revisit the model of announcement distinctness, to determine if there are ways
  to produce a static labeling of messages that does not change based on what
  partition they are in (*this sounds very difficult*).

### Acknowledgements
Two thank-yous in advance go out to Dr. Anderson and Dr. Gersch, for allowing me
to focus on a project that spans topics in both AI and networking, and for their
collective thoughts on the direction of this project.

A special thanks to Spiros and Dimitrios for their assistance in getting started
with BGPMon's data.

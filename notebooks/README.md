# The `notebooks` Directory
This directory contains Jupyter Notebooks relevant to this project, and viewable
on GitHub.

## `BGPDistinct`
This is the main notebook, and one of two major deliverables, for this project.
It contains all data experiments that utilize all major components of this
project. Additionally, the document provides introductory information and
discusses results.

## `Data_Experiments`
This notebook contains initial "playing" with BGPMon data; specifically, a set
of data from one hour on November 1st, 2018 was used. After preprocessing had
been completed, the set included over 250,000 BGP update/announcement messages,
and over 30,000 "unique" events in terms of prefix, prefix mask, and final AS
destination.

The notebook explores different aspects of the data, as well as how
time-interval factors play a role for such BGP messages. Note that this
notebook was written before some major components of the `preprocessing` package
existed, and so some of the original `preprocessing` code can be found inside
the notebook.

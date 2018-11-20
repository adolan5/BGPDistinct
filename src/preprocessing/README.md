# The `preprocessing` Directory
*Responsible for initial parsing of original data for formatting and filtering.*

## Overview
The contents of this directory include code to format and filter original data
from [BGPMon](https://www.bgpmon.io/). The original data comes in the form of
individual JSON objects, but not in an array. That is, the data looks something
like:
```
{One BGP message}
{Another BGP message}
{Another BGP message}
["undefined type" message]
{Another BGP message}
{Another BGP message}
```

However, this is not valid JSON suitable for parsing, and not all BGP messages
are suitable for the purposes of this project.

## Step 1: Formatting for Valid JSON
To convert a dataset to valid JSON, all objects (messages) will be placed into a
JSON array. In other words, opening and closing square brackets will be added to
the top and bottom of a dataset, respectively, and commas will be appropriately
placed. During this step, lines that represent an undefined type message or
other malformed data will be removed.

## Step 2: Filtering by Message Type
Once the data has been transformed to a valid JSON format containing only BGP4
messages, it will be filtered by message content. That is, messages that are not
announcements/advertisements of BGP routes will be eliminated. These messages
can be identified by the fact that they only contain BGP withdrawal content.
The transformed data will be contained in a `list` of `dict`s, and may optionally
be output to its own file. This final data would take a form similar to:
```
[
    {One BGP message},
    {Another BGP message},
    {Another BGP message},
    {Another BGP message},
    {Another BGP message}
]
```

## In Practice: The `preprocessing` Package
This directory defines the `preprocessing` package for BGPDistinct. This package
contains a single module (and class): `DataFormatter`. This class performs the
actions described above. An example of its use can be found [here](/src/doPreprocessing.py).

The transformed data, as described above, is held in an instance variable named
`formatted_data`. `DataFormatter` offers an instance function, `output_data` to
output the transformed data to the file specified.

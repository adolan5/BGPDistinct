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
The transformed data will be contained in a `list` of `dict`s, and may
optionally be output to its own file. This final data would take a form similar
to:
```
[
    {One BGP message},
    {Another BGP message},
    {Another BGP message},
    {Another BGP message},
    {Another BGP message}
]
```

## Step 3: Extraction of Message Components
Initial [exploration](/notebooks/Data_Experiments.ipynb) of the data has
revealed some ways that the raw data should be interpreted and different
features should be extracted. The primary items that are useful to this
application are:
* Timestamp of announcement
* Announcement prefix
* Announcement prefix type (IPv4 or IPv6)
* Announcement mask (e.g., `24` for a /24 prefix)
* Destination AS number (final AS number on an `AS_PATH`

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

#### 3: Aggregation
Initial work with the data also notes a particular caveat- the BGP4
specification allows for the aggregation of multiple prefixes in a single
announcement, in the case that they share the same final destination. For the
purposes of this project, each `(prefix, mask, destination_as)` composite key
will be considered to be a "single" announcement. Therefore, we must process the
data into a format similar to the following:
```json
[
    {
        "time": 1543531407.0,
        "composite": { "type": "0", "prefix1": 258, "prefix2": 772, "mask": 24, "dest": 25 },
        "full_path": [ 8, 1754, 235, 25 ]
    },
    {"..."}
]
```

Our main variables here are those listed above; time is kept separate from the
prefix, mask, and destination tuple for convenience of data experimentation.
Note as well that we also include the full AS path of the announcement, to be
used during labeling routines. Time is also converted to epoch time, for
convenience, and note that the IP prefix is converted into a 64-bit integer (the
IP used in this example is 1.2.3.4).

## In Practice: The `preprocessing` Package
This directory defines the `preprocessing` package for BGPDistinct. This package
contains two modules:
### `DataFormatter`
This class performs the first two actions described above. An example of its use
can be found [here](/src/doPreprocessing.py).

The transformed data, as described above, is held in an instance variable named
`formatted_data`. `DataFormatter` offers an instance function, `output_data` to
output the transformed data to the file specified.

### `DataExtr`
This class is responsible for step 3 of the process listed above. It takes
properly formatted messages from step 2 and creates the final data format that
includes the timestamp, composite key (address-type, prefix, mask, and destination), and the
full path of any one announcement.

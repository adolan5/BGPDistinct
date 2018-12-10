# The `data/` Directory
This directory is meant to contain data, both raw and processed, relevant to
this project.  Note that JSON files are not checked into version control.

## Raw Data
The raw data used for the [preprocessing](/src/preprocessing) step of this
project comes directly from the BGPMon project, in JSON format. For example, a
single BGP message in this format looks like:
```json
{"mrt_header":{"type":16,"subtype":4,"len":96,"timestamp":"2018-10-31T18:00:00-06:00"},"bgp4mp_header":{"peer_as":####,"local_as":####,"address_family":1,"peer_ip":"#.#.#.#","local_ip":"#.#.#.#"},"bgp_header":{"marker":"/////////////////////w==","length":76,"type":2},"bgp_update":{"advertized_routes":[{"prefix":"#.#.#.#","mask":24}],"attrs":{"transitive_bit":true,"as_path":[{"as_seq":[####,####,#####]}],"types":[1,2,7],"aggregator":{"as":#####,"ip":"#.#.#.#"}}}},
```

This data clearly has a number of characteristics that will be of key interest
for this project, including the `timestamp`, `as_path`, and `prefix`.

## Data Used for Experiments
The `get_data` script can be used to download the data used in this project's
experiments. The script assumes that one of `Wget` or `cURL` is installed and
availalble.

By default, a data set that spans 60 seconds is downloaded. The resulting file
is 4MB in size. If desired, a larger dataset that spans an hour can be
downloaded by passing any argument to `get_data`. Take caution, the resulting
file is 160MB in size. The script will abort its operation if the resulting file
(of a static name) already exists.

#!/bin/bash
FILE=raw_one_min_nov15.json
curl -o $FILE http://bgpmon.io/archive/json/routeviews4/updates?start=20181115202200\&end=20181115202300
curl http://bgpmon.io/archive/json/routeviews2/updates?start=20181115202200\&end=20181115202300 >> $FILE

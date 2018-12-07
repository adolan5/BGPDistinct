#!/bin/bash
# Ensure that the file is path independent
FILE=$( dirname $0)/raw_one_min_nov15.json
echo "Outputting file to $FILE"
curl -o $FILE http://bgpmon.io/archive/json/routeviews4/updates?start=20181115202200\&end=20181115202300
curl http://bgpmon.io/archive/json/routeviews2/updates?start=20181115202200\&end=20181115202300 >> $FILE

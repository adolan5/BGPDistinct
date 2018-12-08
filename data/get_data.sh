#!/bin/bash
# Ensure that the file is path independent
FILE=$( dirname $0)/raw_one_min_nov15.json
# Use curl or wget
echo "Outputting file to $FILE"
if [ `command -v curl` ]; then
  echo "Using cURL"
  curl -o $FILE http://bgpmon.io/archive/json/routeviews4/updates?start=20181115202200\&end=20181115202300
  curl -o - http://bgpmon.io/archive/json/routeviews2/updates?start=20181115202200\&end=20181115202300 >> $FILE
else
  echo "No cURL. Defaulting to Wget..."
  wget -O $FILE http://bgpmon.io/archive/json/routeviews4/updates?start=20181115202200\&end=20181115202300
  wget -O - http://bgpmon.io/archive/json/routeviews2/updates?start=20181115202200\&end=20181115202300 >> $FILE
fi

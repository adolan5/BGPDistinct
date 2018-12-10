#!/bin/bash

# This script can be used to download data sets utilized for the BGPDistinct project.
# The minute long data set is assumed when this command is run with no arguments.
# If a single argument is passed, the large dataset is assumed (160MB).
# Requires cURL or Wget

# Ensure that the file is path independent
FILE=$(dirname $0)/raw_one_min_nov15.json
# Assume the small file
FIRST_URL=http://bgpmon.io/archive/json/routeviews4/updates?start=20181115202200\&end=20181115202300
SEC_URL=http://bgpmon.io/archive/json/routeviews2/updates?start=20181115202200\&end=20181115202300

# Use curl or wget
if [ `command -v curl` ]; then
  echo "Using cURL"
  COMMAND="curl -o"
else
  echo "No cURL. Defaulting to Wget..."
  COMMAND="wget -O"
fi

# If the user asked for the large file
if [ $# -gt 0 ]; then
  FILE=$(dirname $0)/raw_large_set_nov1.json
  FIRST_URL=http://bgpmon.io/archive/json/routeviews2/updates?start=20181101000000\&end=20181101010000
  SEC_URL=
fi

# Only fetch if the file does not already exist
if [ -e $FILE ]; then
  echo "File $FILE already exists. Aborting."
  exit 1
fi

echo "Outputting file to $FILE"
$COMMAND $FILE $FIRST_URL
# Account for if there's no second url
if [ -n "$SEC_URL" ]; then
  $COMMAND - $SEC_URL >> $FILE
fi

exit 0

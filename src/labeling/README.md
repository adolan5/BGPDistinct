# The `labeling` Package
*For labeling formatted and extracted BGP data with some semblance of ease.*

## Motivation and Goals
This package will be used to label BGP data collected from the BGPMon project in
terms of being "distinct" or not. There are some important considerations to be
made regarding this issue:
* Order of appearance largely defines "distinctness," that is, a message is only
  repeated when it follows some other message that came before it.
  * Therefore, order must be preserved when reading data.
* For a reasonable time frame, a majority of (prefix, mask, destination)
  composite keys only appear once, somewhat limiting how many messages need to
  be double-checked.

The ideal use of this package would be for a user to provide a set of properly
formatted data, output by the preprocessing step, and use it to label the
class of each message in that data set. The user would only be prompted to
check if a message matches one that has already been seen if its (prefix, max,
final destination) tuple has already been seen. In such a case, the user must
decide if the message is distinct or not, based on:
* The timestamp of the message.
* The full `AS_PATH` of the message, and if it is similar enough to any distinct
  messages that matched on composite key.

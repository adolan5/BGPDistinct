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

## Need to Reevaluate
The loosely defined process above is somewhat idealistic, and holds key
drawbacks:
* How confident can the user be in her knowledge that two BGP messages are from
  different events? How prone is she to making mistakes?
* How many entries can feasibly be labeled by hand by a user?
  * Even if the mode for the number of times a message composite key appears is
    1, some composites may appear hundreds or even thousands of times; does the
    user need to check every single conflict?

### How can this Model be Improved?
Let's reconsider how we determine if two different messages are distinct, vs. if
they belong to the same logical event.

If two messages *A* and *B* belong to the same event, then they must share the
same prefix, mask, and destination-AS composite key- this was established
before.

However, they may still share this information and be members of separate
events. In this case, the best course of action is to evaluate their
"distinctness" based on:
* Temporal difference
  * How much temporal difference?
  * We're working with mere seconds for the purposes of this project, so this
    category is less likely to have a great impact.
* Differences in `AS_PATH`
  * *How* different are the paths advertized by each router? This could be an
    indication of very different origins.

With these items in mind, how can data be labeled without having to have the
user check every single conflict of composite keys? The user should only be
prompted for a double-check when the system has determined that the
"distinct-ness" of a message is truly ambiguous, given the two pieces of
information above. Therefore, the `AS_PATH` of each message will be compared in
the case of a conflict. If they are significantly different, their distinctness
will be considered ambiguous, and user intervention will take place. This will
hopefully limit the number of analyses required by the user.

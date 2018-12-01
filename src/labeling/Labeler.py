class Labeler:
    """The Labeler class.
    Provides the functionalities to label a set of BGP messages that have been
    formatted and had their features extracted. Provides utilities for a user
    to specify whether a message is distinct or not.
    """
    def __init__(self, data):
        """Constructor.
        Starts labeling routine of the provided data.
        Args:
        data (list): The extracted and formatted BGP data to be labeled.
        """
        # Keep internal reference to data; will be modified in place
        self._data = data

        # Initialize basic tracking
        # This struct maps a (prefix, mask, destination) to a list of messages that were
        # marked distinct, whether automatically (first message), or by the user
        self._seen_distincts = {}

        # Kick off labeling routine
        print('There are {} messages to be labeled'.format(len(data)))
        self._do_labeling()

    def _do_labeling(self):
        """Main labeling subroutine.
        If a message composite key is encountered for the first time, that
        message is considered distinct.
        If a message matches a composite key already seen, ask the user if
        it should be considered distinct, listing one or more of the most
        recent other matches that were marked as distinct.
        Modifies the data dict in place, setting the "distinct" field to
        either 1 (distinct) or 0 (duplicated).
        """
        pass

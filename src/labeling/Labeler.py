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
        # For every message...
        for mess in self._data:
            # Get composite key, see if it has been found before
            comp_key = tuple(mess.get('composite').values())
            if comp_key in self._seen_distincts:
                # Prompt the user for input
                self._prompt_user(mess)
            # Case where the message is the first of its kind; start a new list
            else:
                self._seen_distincts[comp_key] = [mess]
                # Classify message as distinct
                mess['distinct'] = 1

    def _prompt_user(self, message):
        """Prompt the user for if a message is unique or not.
        Precondition: The message in question has been determined to exist in the
        _seen_distincts keyset. This function determines whether or not the provided
        message should be considered distinct.
        """
        pass

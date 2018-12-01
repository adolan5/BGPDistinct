class Labeler:
    """The Labeler class.
    Provides the functionalities to label a set of BGP messages that have been
    formatted and had their features extracted. Provides utilities for a user
    to specify whether a message is distinct or not.
    """
    # Threshold for ambiguity
    PATH_THRESH = 0.5

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
        If a message matches a composite key already seen, check if the
        "distinct-ness" is ambiguous. If so, ask the user for intervention.
        Modifies the data dict in place, setting the "distinct" field to
        either 1 (distinct) or 0 (duplicated).
        """
        # For every message...
        for i, mess in enumerate(self._data):
            # Get composite key, see if it has been found before
            comp_key = tuple(mess.get('composite').values())
            if comp_key in self._seen_distincts:
                # Check for ambiguity, prompt user if necessary
                self._check_ambig(mess, comp_key, i)
            # Case where the message is the first of its kind; start a new list
            else:
                self._seen_distincts[comp_key] = [mess]
                # Classify message as distinct
                mess['distinct'] = 1

    def _check_ambig(self, message, comp, i):
        """Check if a conflicted message is ambiguous and requires user intervention.
        A message is compared to the most recent distinct message that shares its
        composite key. If the AS_PATH (full_path) elements of the messages are
        significantly different from one another, then user intervention ensues.
        TODO: Consider temporal difference here as well, for longer-spanning data
        sets.
        """
        # Most recent distinct message with matching composite key
        most_recent = self._seen_distincts.get(comp)[-1]

        # Condition: Do the contents of the full AS_PATH of each message differ
        # by over 50%? (no duplicates)
        curr_path = set(message.get('full_path'))
        recent_path = set(most_recent.get('full_path'))

        diff_left = len(curr_path - recent_path) / len(curr_path.union(recent_path))
        diff_right = len(recent_path - curr_path) / len(curr_path.union(recent_path))

        # If threshold met, prompt user, let them decide
        if diff_left > self.PATH_THRESH or diff_right > self.PATH_THRESH:
            self._prompt_user(message, comp, i)
        # Else, set distinct flag to duplicate (0)
        else:
            message['distinct'] = 0

    def _prompt_user(self, message, comp, i):
        """Prompt the user for if a message is unique or not.
        Precondition: The message in question has been determined to exist in the
        _seen_distincts keyset. This function determines whether or not the provided
        message should be considered distinct.
        Args:
        message (dict): The message in question.
        comp (tuple): The composite key of the message in question.
        i (int): The index of the message in question.
        """
        most_recent = self._seen_distincts.get(comp)[-1]

        print('\n')
        print('Message {} of {} ({:.2%})'.format(i + 1, len(self._data), ((i + 1) / len(self._data))))
        print('Time diff in sec: {}'.format(message.get('time') - most_recent.get('time')))
        print('Message ambiguity. Most recent distinct version\'s path is:')
        print('Path: {}'.format(most_recent.get('full_path')))
        print('Message in question:')
        print('Path: {}'.format(message.get('full_path')))
        # Main input
        reply = input('Do these belong to the same event? [y]/n: ')

        if reply.lower() in ['', 'y', 'yes']:
            print('Marked as duplicate')
            # This is in place; will update message at top level
            message['distinct'] = 0
        else:
            print('Marked as distinct')
            # Set flag and append to most recently seen
            message['distinct'] = 1
            self._seen_distincts[comp].append(message)

            print(self._seen_distincts.get(comp))

class Labeler:
    """The Labeler class.
    Provides the functionalities to label a set of BGP messages that have been
    formatted and had their features extracted. Provides utilities for a user
    to specify whether a message is distinct or not.
    """
    # Thresholds for ambiguity
    PATH_THRESH = 0.5
    # Seconds that messages are apart
    TIME_THRESH = 5

    def __init__(self, data, simple=True, debug=False):
        """Constructor.
        Starts labeling routine of the provided data.
        Args:
        data (list): The extracted and formatted BGP data to be labeled.
        simple (bool): Whether or not to perform labeling in a simple way;
            the alternative being the labeling routine that involves user
            intervention via the prompt.
        debug (bool): Whether or not to output some debug information about labeling.
        """
        # Keep internal reference to data; will be modified in place
        self._data = data
        self._num_mess = len(data)

        # Keep labeling simple, or ask for user intervention in times of ambiguity?
        self._simple = simple
        self._debug = debug

        # Initialize basic tracking
        # This struct maps a (prefix, mask, destination) to a list of messages that were
        # marked distinct, whether automatically (first message), or by the user
        self._seen_distincts = {}

        # Metrics for instance
        self.num_ambig = 0
        self.num_prompt = 0

        # Kick off labeling routine
        print('There are {} messages to be labeled'.format(len(data)))
        self._do_labeling()

    def _do_labeling(self):
        """Main labeling subroutine.
        If a message composite key is encountered for the first time, that
        message is considered distinct.
        If a message matches a composite key already seen, perform the following:
        * If the Labeler is in "simple" mode, simply mark the message as
          duplicated.
        * If the Labeler is in "advanced" mode, check if the "distinct-ness" is
          ambiguous. If so, ask the user for intervention.
        In either case, this routine modifies the data dict in place, setting
        the "distinct" field to either 1 (distinct) or 0 (duplicated).
        """
        # For every message...
        for i, mess in enumerate(self._data):
            if(self._debug):
                print('Message {} of {} ({:.2%})'.format(i + 1, self._num_mess, ((i + 1) / self._num_mess)), end='\r')
            # Get composite key, see if it has been found before
            comp_key = tuple(mess.get('composite').values())
            conflict = (comp_key in self._seen_distincts)

            # Act appropriately for the configuration and situation
            if conflict and self._simple:
                # Case where we simply assume duplicates for subsequent messages
                mess['distinct'] = 0
            elif conflict:
                # Check for ambiguity, prompt user if necessary (Advanced labeling)
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
        self.num_ambig += 1
        # Most recent distinct message with matching composite key
        most_recent = self._seen_distincts.get(comp)[-1]
        time_diff = message.get('time') - most_recent.get('time')

        # Condition: Do the contents of the full AS_PATH of each message differ
        # by over 50%? (no duplicates)
        curr_path = set(message.get('full_path'))
        recent_path = set(most_recent.get('full_path'))

        diff_left = len(curr_path - recent_path) / len(curr_path.union(recent_path))
        diff_right = len(recent_path - curr_path) / len(curr_path.union(recent_path))

        # If threshold met, prompt user, let them decide
        if (diff_left > self.PATH_THRESH or diff_right > self.PATH_THRESH) and time_diff > self.TIME_THRESH:
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
        self.num_prompt += 1
        most_recent = self._seen_distincts.get(comp)[-1]
        # Get paths with no duplicates while preserving order
        seen = set()
        # Short-circuited
        recent_path = [a for a in most_recent.get('full_path') if not (a in seen or seen.add(a))]
        seen.clear()
        curr_path = [a for a in message.get('full_path') if not (a in seen or seen.add(a))]

        print('\n')
        print('Time diff in sec: {}'.format(message.get('time') - most_recent.get('time')))
        print('Message ambiguity. Most recent distinct version\'s path is:')
        print('Simplified path: {}'.format(recent_path))
        print('Message in question:')
        print('Simplified path: {}'.format(curr_path))
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
        print('\n')

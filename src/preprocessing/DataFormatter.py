import fileinput
import json

"""The DataFormatter class.
Used to transform raw BGPMon data into properly formatted JSON. The
transformed data can then be output to a file, or simply retrieved from
memory.
"""
class DataFormatter:
    def __init__(self, filename):
        """Constructor.
        Loads a file and calls out to perform transformations.
        Args:
        filename (str): The path to the file from which to read the original
            raw data.
        """
        # Call out to transform data
        self.formatted_data = self._transform_data(filename)


    def _transform_data(self, filename):
        """Transform raw data from a BGPMon file to properly formatted data.
        Also remove BGP updates that are only withdrawal messages.
        Args:
        raw_data (str): The path to the input file.
        Returns:
        A list of BGP4 specification JSON objects (dictionaries) with relevant fields.
        """
        formatted = []

        # Read the file line by line, keep what we want
        f = fileinput.input(filename)
        for line in f:
            # In the case of an "unknown type" or a malformed line, we skip it
            try:
                candidate_line = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Now validate that this object contains the keys we want, and that it is the
            # proper BGP message (announcement/advertisement)
            if not self._validate_message(candidate_line):
                continue

            # All checks have passed; append to returned structure
            formatted.append(candidate_line)

        return formatted

    def _validate_message(self, message):
        """Validate that a candidate BGP message is of the correct type and
        has the correct keys relevant to analysis.
        Args:
        message (dict): The candidate BGP message to validate.
        Returns:
        True if the message is of the correct type and has the proper keys
            for analysis.
        """
        # Validate that this message is an advertisement/announcement
        if 'advertized_routes' not in message.get('bgp_update', {}):
            return False
        # Validate that the BGP AS_PATH doesn't contain an AS_SET
        path_segments = message.get('bgp_update').get('attrs', {}).get('as_path', [])
        # If any of the path segments contain an 'as_set' field, invalidate this message
        if any([seg.get('as_set') for seg in path_segments]):
            return False

        # Checks pass; return True
        return True

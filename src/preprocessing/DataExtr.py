import json
import socket

class DataExtr:
    """The DataExtr class.
    Responsible for extracting key features from formatted BGP messages.
    """
    def __init__(self, raw_data):
        """Constructor.
        Performs feature extraction on the provided raw data.
        Args:
        raw_data (list of dict): The formatted BGP data from which to extract features.
        """
        self.extracted_data = self._do_extraction(raw_data)

    def output_data(self, filename):
        """Output extracted features."""
        with open(filename, 'w') as f:
            json.dump(self.extracted_data, f)

    def _do_extraction(self, data):
        """Main extraction routine.
        Extracts relevant features from formatted BGP data.
        Args:
        data (list of dict): The original formatted BGP data.
        Returns:
        A new list containing extracted features for each single-prefix BGP announcement.
        """
        transformed_data = []
        for message in data:
            # Timestamp and destination are the same for all prefix-messages in
            # the case of an aggregated message
            tstamp = message.get('mrt_header').get('timestamp')
            dest = message.get('bgp_update').get('attrs').get('as_path')[-1].get('as_seq')[-1]

            # Reconstruct the full AS path, accounting for if it has multiple segments
            full_path = [r for p in message.get('bgp_update').get('attrs').get('as_path') for r in p.get('as_seq')]

            # Create new message dict for each single-prefix announcement
            # Note that the prefix is also converted to a number here
            # TODO: Is such a conversion problematic memory wise? IPv6 is 128-bit...
            transformed_data.extend(
                    [{'time': tstamp, 'composite': {
                        'prefix': self._conv_address(r.get('prefix')),
                        'mask': r.get('mask'),
                        'dest': dest
                    },
                    'full_path': full_path
                    } for r in message.get('bgp_update').get('advertized_routes')]
            )
        # Also sort data by time, for good measure
        return sorted(transformed_data, key=lambda s: s.get('time'))

    def _conv_address(self, addr):
        """Convert an IP address to an integer.
        Accounts for both IPv4 and IPv6.
        """
        # Determine type
        if ':' in addr:
            addr_type = socket.AF_INET6
        else:
            addr_type = socket.AF_INET
        # Perform conversion
        return int.from_bytes(socket.inet_pton(addr_type, addr), 'big')

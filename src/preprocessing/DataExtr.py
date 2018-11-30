import json

"""The DataExtr class.
Responsible for extracting key features from formatted BGP messages.
"""
class DataExtr:
    def __init__(self, raw_data):
        """Constructor.
        Performs feature extraction on the provided raw data.
        Args:
        raw_data (list of dict): The formatted BGP data from which to extract features.
        """
        self.final_data = self._do_extraction(raw_data)

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
            transformed_data.extend(
                    [{'time': tstamp, 'composite': {
                        'prefix': r.get('prefix'),
                        'mask': r.get('mask'),
                        'dest': dest
                    },
                    'full_path': full_path
                    } for r in message.get('bgp_update').get('advertized_routes')]
            )
        return transformed_data

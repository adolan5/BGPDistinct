import json
import ipaddress

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
            # Note that the prefix is also converted to a number here, and its
            # type is captured (0 for v4 and 1 for v6)
            for pref in message.get('bgp_update').get('advertized_routes'):
                prefix1, prefix2 = self._conv_address(pref.get('prefix'))

                # Create composite key for this prefix
                comp = {'type': 0 if ':' not in pref.get('prefix') else 1,
                        'prefix1': prefix1,
                        'prefix2': prefix2,
                        'mask': pref.get('mask'),
                        'dest': dest}

                transformed_data.append({'time': tstamp, 'composite': comp, 'full_path': full_path})
        # Also sort data by time, for good measure
        return sorted(transformed_data, key=lambda s: s.get('time'))

    def _conv_address(self, addr):
        """Convert an IP address to two integers.
        Accounts for both IPv4 (16-bit max int) and IPv6 (64 bit max int).
        Returns:
        A tuple containing two signed integers representing the first half
            of the address and the second half of the address.
        """
        # Determine type and perform conversion
        # IPv6
        if ':' in addr:
            # Expand to full IP
            full_ip = ipaddress.ip_address(addr).exploded
            octets = [o for o in full_ip.split(':')]
            # 64 bit integers for each half
            first_half, second_half = (int(''.join(octets[:4]), 16), int(''.join(octets[4:]), 16))
        # IPv4
        else:
            octets = [int(o) for o in addr.split('.')]
            first_half = sum([(256 ** i) * o for i, o in enumerate(reversed(octets[:2]))])
            second_half = sum([(256 ** i) * o for i, o in enumerate(reversed(octets[2:]))])

        # Ensure that numbers are signed, they may be large enough- 2's complement
        # TODO: This smells very hacky
        if first_half > 0x7FFFFFFFFFFFFFFF:
            first_half = -((first_half ^ 0xFFFFFFFFFFFFFFFF) + 1)
        if second_half > 0x7FFFFFFFFFFFFFFF:
            second_half = -((second_half ^ 0xFFFFFFFFFFFFFFFF) + 1)

        return (first_half, second_half)

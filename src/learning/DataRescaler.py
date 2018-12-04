import socket
import numpy as np
import math
from copy import deepcopy

class DataRescaler:
    """The DataRescaler class.
    Responsible for converting and rescaling data points for use in AI implementations.
    """
    def __init__(self, raw_data):
        """Constructor.
        Performs data alterations on the provided raw data.
        Args:
        raw_data (list): The raw BGP data that has been formatted and had
            its relevant features extracted (via DataExtr).
        """
        self.scaled_data = self._do_rescaling(raw_data)

    def _do_rescaling(self, orig_data):
        """Main rescaling routing.
        This function converts ip addresses to numbers (and scales them down
        appropriately), and standardizes all data points to have 0 mean and
        1 standard deviation.
        Args:
        orig_data (list): The original data to be transformed.
        """
        # Start with a new copy
        data = deepcopy(orig_data)

        # Start with addresses; alter them on the fly
        for m in data:
            m.get('composite')['prefix'] = self._conv_address(m.get('composite').get('prefix'))

        comps = [d.get('composite') for d in data]

        # Now for rescaling of all relevant fields (not full_path)
        # Take advantage of pass-by-reference
        self._rescale(data, 'time')
        self._rescale(comps, 'prefix')
        self._rescale(comps, 'mask')
        self._rescale(comps, 'dest')

        return data

    def _rescale(self, data, key):
        """Rescale a single data point in a structure.
        Args:
        data (list of dict): The structure to modify in place.
        key (str): The key of the element to be modified.
        """
        # Get mean and standard devation
        all_vals = [v.get(key) for v in data]
        mean = np.mean(all_vals)
        std = np.std(all_vals)

        # Now update in place
        for single in data:
            single[key] = (single.get(key) - mean) / std

    def _conv_address(self, addr):
        """Convert an IP address to an integer.
        Accounts for both IPv4 (16-bit max int) and IPv6 (64 bit max int).
        Returns:
        An integer representation of the provided IP address, which has been "squashed"
            via a log.
        """
        # Determine type and perform conversion
        # IPv6
        if ':' in addr:
            addr_type = socket.AF_INET6
        # IPv4
        else:
            addr_type = socket.AF_INET

        converted = int.from_bytes(socket.inet_pton(addr_type, addr), 'big')

        # Squash possibly large numbers a bit, smallest possible prefix is 0
        # so we also add 1
        return math.log(converted + 1)

#!/usr/bin/env python3
from preprocessing import DataFormatter as df
import time

# Create a new df and output some data to the console
start = time.time()
# my_df = df('../data/raw_large_set.json')
my_df = df('../data/nov_15_one_hour.json')
end = time.time()
print('Preprocessing took {:.3f}'.format(end - start))

# Output some of the data
print('Length of data is {:d}. First 5 entries:'.format(len(my_df.formatted_data)))
print(my_df.formatted_data[:5])
print()

# Output some metrics captured from the data
print('Number of malformed/unknown messages: {:d}'.format(my_df.num_malformed))
print('Number of withdraw-only messages: {:d}'.format(my_df.num_withdraws))
print('Number of messages containing AS_SET: {:d}'.format(my_df.num_as_set))
print('Total number of BGP messages: {:d}'.format(my_df.get_num_messages()))
print('Total number of BGP announcements: {:d}'.format(len(my_df.formatted_data)))

# Output the data to a new file
# my_df.output_data('../data/formatted_large_set.json')
my_df.output_data('../data/formatted_nov_15.json')

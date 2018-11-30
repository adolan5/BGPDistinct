#!/usr/bin/env python3
from preprocessing import DataFormatter as df
from preprocessing import DataExtr as extr
import time

print('Steps 1 and 2: Formatting and filtering data')
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

# Output the formatted data to a new file
# my_df.output_data('../data/formatted_large_set.json')
# my_df.output_data('../data/formatted_nov_15.json')

print('\nStep 3: Feature extraction')
# Extract features from data
start = time.time()
my_extr = extr(my_df.formatted_data)
end = time.time()
print('Feature extraction took {:.3f}'.format(end - start))

# Statistics about extracted data
print('Number of single-prefix messages: {:d}'.format(len(my_extr.final_data)))
print('Number of "distinct" composites: {:d}'.format(len(set([m.get('composite').values() for m in my_extr.final_data]))))
print('First 5 formatted messages:')
print(my_extr.final_data[:5])

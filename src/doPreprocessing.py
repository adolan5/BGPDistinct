#!/usr/bin/env python3
from preprocessing import DataFormatter as df
import time

# Create a new df and output some data to the console
start = time.time()
my_df = df('../data/raw_small_set.json')
end = time.time()
print('Preprocessing took {:.3f}'.format(end - start))

# Output some of the data
print('Length of data is {:d}. First 5 entries:'.format(len(my_df.formatted_data)))
print(my_df.formatted_data[:5])

# Output the data to a new file
my_df.output_data('../data/transformed_small_set.json')

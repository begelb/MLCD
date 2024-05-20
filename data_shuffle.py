import pandas as pd

file_name = 'data/periodic_3_labels/data_periodic_shuffled_balanced.csv'
new_file_name = 'data/periodic_3_labels/data_periodic_shuffled_balanced_shuffled_again.csv'

df = pd.read_csv(file_name, header=None)
ds = df.sample(frac=1)
ds.to_csv(new_file_name, index = False, header=False)
import pandas as pd

dim = 5

file_name = f'data/ellipsoidal_larger_domain_{dim}d/train.csv'
new_file_name = f'data/ellipsoidal_larger_domain_{dim}d/new_train.csv'
test_file_name = f'data/ellipsoidal_larger_domain_{dim}d/test.csv'



df = pd.read_csv(file_name, header=None)

train_set = df.sample(frac=0.8, random_state=42) 
  
# Dropping all those indexes from the dataframe that exists in the train_set 
test_set = df.drop(train_set.index)

# ds = df.sample(frac=1)
train_set.to_csv(new_file_name, index = False, header=False)
test_set.to_csv(test_file_name, index = False, header=False)
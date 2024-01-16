import pandas as pd


df_train_url = 'https://raw.githubusercontent.com/begelb/attractor-id-data/main/system1/test.csv?token=GHSAT0AAAAAACMNABDWXNDT5FWRIZJ7WVY2ZNG2H3A'

df_train = pd.read_csv(df_train_url, header = None)

dimension = 3
print(df_train.iloc[0:1])
print(df_train.iloc[1:2])

print(df_train.iloc[1:2].values.tolist()[0])

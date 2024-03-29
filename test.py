import pandas as pd
import pickle

with open(f'data2/system10/exp_info.pickle', 'rb') as handle:
    exp_info = pickle.load(handle)

print(exp_info)
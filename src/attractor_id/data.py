import csv
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd

class DatasetFromPandas(Dataset):
    def __init__(self, dataframe):
        self.d = len(dataframe.columns) - 1

        self.dataframe = dataframe

        column_dict = dict()
        for i in dataframe.columns:
            column_dict[i] = dataframe.iloc[:,i]

        self.column_dict = column_dict

        data = []
        for i in range(len(dataframe)):
            data.append(dataframe.iloc[i:i+1].values.tolist()[0])
        self.data = data

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        point = torch.zeros(self.d)
        for i in range(0, self.d):
            point[i] = float(self.data[idx][i])
        label = int(round(self.column_dict[self.d][idx]))
        label = torch.ones(1, dtype=torch.float32) * label

        return point, label

class MyDataset(Dataset):
    def __init__(self, dimension, annotations_file):
        self.d = dimension
        self.data = np.loadtxt(annotations_file, delimiter=',', dtype=np.float32)
        self.features = torch.tensor(self.data[:, :self.d], dtype=torch.float32)
        self.labels = torch.tensor(self.data[:, self.d].round(), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# class MyDataset(Dataset):
#     def __init__(self, d, annotations_file):
#         self.d = d
#         with open(annotations_file, newline='') as f:
#             reader = csv.reader(f)
#             self.data = list(reader)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         data_point = self.data[idx]
#         point = torch.zeros(self.d)
#         for i in range(0, self.d):
#             point[i] = float(data_point[i])
#         label = int(round(float(data_point[self.d])))
#         label = torch.ones(1, dtype=torch.float32) * label
#         return point, label
    
def data_set_up(config, using_pandas = False):
    d = config.dimension

    if using_pandas:
        df_train_url = config.train_url
        df_test_url = config.test_url
        df_train = pd.read_csv(df_train_url, header = None)
        df_test = pd.read_csv(df_test_url, header = None)

        train_dataset = DatasetFromPandas(df_train)
        test_dataset = DatasetFromPandas(df_test)

        batch_size = config.batch_size

        train_data = DatasetFromPandas(df_train).data
        test_data = DatasetFromPandas(df_test).data
        
    else:
        train_data_file = config.train_data_file
        test_data_file = config.test_data_file

        train_dataset = MyDataset(d, train_data_file)
        test_dataset = MyDataset(d, test_data_file)

        train_data = train_dataset.data

        test_data = test_dataset.data

        batch_size = config.batch_size

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)
    figure_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

    return train_data, test_data, train_dataloader, test_dataloader, figure_dataloader

def point_inside_domain(point, data_bounds_list, d):
    tolerance = 1e-4
    pt_inside_domain_Bool_tracker = []
    for i in range(d):
        if float(point[i]) + tolerance < data_bounds_list[i][0] or float(point[i]) - tolerance > data_bounds_list[i][1]:
            pt_inside_domain_Bool_tracker.append(False)
        else:
            pt_inside_domain_Bool_tracker.append(True)
    if all(pt_inside_domain_Bool_tracker):
        return True
    else:
        return False
    
def convert_data_to_tensors(data, d):
    
    data_tensor_list = np.array(data[:, :d], dtype=np.float32)

    return data_tensor_list
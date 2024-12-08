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
    def __init__(self, d, annotations_file):
        self.d = d
        with open(annotations_file, newline='') as f:
            reader = csv.reader(f)
            self.data = list(reader)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        point = torch.zeros(self.d)
        for i in range(0, self.d):
            point[i] = float(data_point[i])
        label = int(round(float(data_point[self.d])))
        label = torch.ones(1, dtype=torch.float32) * label
        return point, label
    
class DatasetForOneVRest(Dataset):
    # spotlight class is the class which is highlighted (the "one") versus the other classes 
    def __init__(self, spotlight_class, d, annotations_file):
        self.d = d
        self.spotlight_class = spotlight_class
        with open(annotations_file, newline='') as f:
            reader = csv.reader(f)
            self.data = list(reader)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        point = torch.zeros(self.d)
        for i in range(0, self.d):
            point[i] = float(data_point[i])
        original_label = int(round(float(data_point[self.d])))

        # if the label in the dataset corresponds to the spotlight_class, then it is given a final label of 1
        # otherwise, it is given a final label of 0
        if original_label == self.spotlight_class:
            label = 1
        else:
            label = 0
        label = torch.ones(1, dtype=torch.float32) * label
        return point, label
    
class EnsembleData():
    def __init__(self, config):
        d = config.dimension

        batch_size = config.batch_size

        train_dataloaders_dict = dict()
        test_dataloaders_dict = dict()

        train_data_file = config.train_data_file
        test_data_file = config.test_data_file

        for spotlight_label in range(config.num_labels):
            train_dataset = DatasetForOneVRest(spotlight_label, d, train_data_file)
            test_dataset = DatasetForOneVRest(spotlight_label, d, test_data_file)

            train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)

            train_dataloaders_dict[spotlight_label] = train_dataloader
            test_dataloaders_dict[spotlight_label] = test_dataloader
            
        all_data_train = MyDataset(d, train_data_file)
        all_data_test = MyDataset(d, test_data_file)

        all_train_data = all_data_train.data # do this function where it is needed instead

        all_test_dataloader = DataLoader(all_data_test, batch_size = batch_size, shuffle=True)
        figure_dataloader = DataLoader(all_data_train, batch_size = batch_size, shuffle = False)

        self.train_dataloaders_dict = train_dataloaders_dict
        self.test_dataloaders_dict = test_dataloaders_dict
        self.all_test_dataloader = all_test_dataloader
        self.figure_dataloader = figure_dataloader
        self.all_train_data = all_train_data

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

      #  if len(train_data)%10 == 0:
        batch_size = config.batch_size
       # else:
        #    batch_size = 1000

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)
    figure_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

    return train_data, test_data, train_dataloader, test_dataloader, figure_dataloader
    
def convert_data_to_tensors(data, d):
    data_tensor_list = []
    for l in range(0, len(data)):
        data_point = data[l]
        data_tensor = np.zeros(d)
        for k in range(0, d):
            data_tensor[k] += float(data_point[k])
        data_tensor_list.append(data_tensor)
    return data_tensor_list
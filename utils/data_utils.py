import csv
import torch
from torch.utils.data import DataLoader, Dataset

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
    
def data_set_up(config):
    train_data_file = config.train_data_file
    test_data_file = config.test_data_file

    with open(train_data_file, newline='') as f:
        reader = csv.reader(f)
        train_data = list(reader)

    batch_size = len(train_data)//10

    d = config.dimension

    train_dataset = MyDataset(d, train_data_file)
    test_dataset = MyDataset(d, test_data_file)

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)
    figure_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)

    return train_data, train_dataloader, test_dataloader, figure_dataloader
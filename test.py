from src.attractor_id.data import DatasetForOneVRest, MyDataset
from src.attractor_id.config import configure
from src.attractor_id.data import data_set_up
from src.attractor_id.network import Regression_Cubical_Network_One_Nonlinearity

system = 'ellipsoidal_larger_domain_5d'
config_fname = f'config/{system}.txt'
config = configure(config_fname)

init_shared_weight_matrix = nn.Parameter(torch.eye(config.dimension), requires_grad=True)

#train_data_file = 'data/ellipsoidal_4d/test.csv' #config.train_data_file
train_data_file = config.train_data_file
dataset1 = MyDataset(config.dimension, train_data_file)


# produce plots of all two-dimensional projections of the data, with points colored according to label

import matplotlib.pyplot as plt
import numpy as np
import torch

# Extract data and labels from the dataset
# data = torch.stack([dataset1[i][0] for i in range(len(dataset1))]).numpy()  # Convert to numpy array
# labels = torch.cat([dataset1[i][1] for i in range(len(dataset1))]).numpy()  # Flatten labels to 1D array

# print the max and min of the data in each column

# for i in range(data.shape[1]):
#     print(f"max of column {i}: {np.max(data[:, i])}")
#     print(f"min of column {i}: {np.min(data[:, i])}")

# # print percentage of each label
# for i in range(4):
#     print(f"percentage of label {i}: {np.mean(labels == i)}")

# load model
model_file_name = '2-model.pth'
train_data, test_data, train_dataloader, test_dataloader, figure_dataloader = data_set_up(config, using_pandas = False)
batch_size = config.batch_size

N = 20

model = Regression_Cubical_Network_One_Nonlinearity(N, 1, config)
model.load_state_dict(torch.load(model_file_name))

exit()

# subsample 1000 points
data = data[:1000]
labels = labels[:1000]

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# choose i to be a random number between 0 and 3
i = 0
j = 1
ax.scatter(data[:, i], data[:, j], c=labels, cmap='tab10', alpha=0.7)
ax.set_xlabel(f'x{i}')
ax.set_ylabel(f'x{j}')
plt.show()


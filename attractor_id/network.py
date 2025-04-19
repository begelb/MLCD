import torch
from torch import nn
import os

def get_batch_size(train_data, percentage):
    return int(len(train_data)*percentage)

def get_optimizer(config, model):
    optimizer_choice = config.optimizer_choice
    learning_rate = config.learning_rate
    if optimizer_choice == 'Adam':
        return torch.optim.Adam(model.parameters(), learning_rate)
    elif optimizer_choice == 'Adagrad':
        return torch.optim.Adagrad(model.parameters(), learning_rate)
    
class Regression_Cubical_Network_One_Nonlinearity(nn.Module):
    def __init__(self, N, batch_size, config):
        super(Regression_Cubical_Network_One_Nonlinearity, self).__init__()
        # N is the width of the first hidden layer
        self.N = N
        # d is the dimension of the input
        d = config.dimension
        self.group_size = N//d
        num_labels = config.num_labels
        data_bounds_list = config.data_bounds
        self.threshold_prediction = config.threshold_prediction
        self.threshold = float(num_labels - 1)

        # shared_weight_matrix is a d x d matrix with rows made up of the weights that are each shared up to a constant multiple (weight_cofficients) among a set of N//d nodes of the hidden layer
        # shared_weight_matrix is initialized as a d x d identity matrix
        self.shared_weight_matrix =  nn.Parameter(torch.eye(d), requires_grad=True)
        # register shared_weight_matrix so that it can be accessed using this name at any point
        self.register_parameter('shared_weight_matrix', self.shared_weight_matrix)

        # weight coefficients are the coefficients that relate the hidden layer weights to the shared_weight_matrix
        # there is one coefficient for each node in the hidden layer
        # if weak_weight_share = True, then the network has trainable weight parameters, so requires_grad below is set to True
        self.weight_coefficients = nn.Parameter(torch.ones(self.N), requires_grad = config.weak_weight_share)
        self.register_parameter('weight_coefficients', self.weight_coefficients)

        # assign each node in the hidden layer a key between 0 and d - 1, with N // d nodes being assigned each key
        hidden_node_keys = torch.zeros(N, dtype=torch.long)
        group_size = N // d
        for i in range(d):
            hidden_node_keys[i*group_size:(i+1)*group_size] = i

        # initialize the biases of the hidden nodes of key = i via a uniform distribution over the interval that makes up the data domain in i-th dimension
        bias_initial_lower_bounds = torch.zeros(N)
        bias_initial_upper_bounds = torch.zeros(N)
        for i in range(d):
            bias_initial_lower_bounds[hidden_node_keys == i] = torch.tensor(data_bounds_list[i][0], dtype=torch.float)
            bias_initial_upper_bounds[hidden_node_keys == i] = torch.tensor(data_bounds_list[i][1], dtype=torch.float)
        
        # each bias is the negation of a hyperplane offset, which is why the next line might look unnatural
        self.bias = nn.Parameter(torch.rand(self.N)*(bias_initial_lower_bounds - bias_initial_upper_bounds) - bias_initial_lower_bounds)
        self.register_parameter('bias', self.bias)

        # define and initialize the unconstrained weights of the neural network, namely the output layers
        self.output_layer = nn.Linear(self.N, 1, bias = True)

        # define the activation function of the network
        self.hardtanh1 = nn.Hardtanh(min_val = 0, max_val = 1)

        self.flatten = nn.Flatten()


    def forward(self, x):
        
        unscaled_weight_matrix = self.shared_weight_matrix.repeat_interleave(self.group_size, dim=0)

        weight_matrix = unscaled_weight_matrix * self.weight_coefficients.view(-1, 1)

        # apply weight matrix and bias vector
        hidden_layer_before_hardtanh = torch.matmul(x, weight_matrix.T) + self.bias
        layer1_output = self.hardtanh1(hidden_layer_before_hardtanh)

        # apply final layer weights 
        output = self.output_layer(layer1_output)

        if self.threshold_prediction:
            return torch.clamp(output, min=0.0, max = self.threshold)
        else:
            return output

def load_model(N, system, config, batch_size, example_index=0):
    cube_reg_model = Regression_Cubical_Network_One_Nonlinearity(N, batch_size, config)
    cube_reg_model.load_state_dict(torch.load(f'output/models/{system}/{example_index}-model.pth'))
    return cube_reg_model

def get_model_parameters(model):
    for i, param in enumerate(model.parameters()):
        param.requires_grad = False # This command means to stop keeping track of the computational graphs used for backpropagation
    biaslist = model.bias
    shared_weight_matrix = model.shared_weight_matrix
    weight_coefficients = model.weight_coefficients
    parameter_dict = dict()
    parameter_dict["biaslist"] = biaslist
    parameter_dict["shared_weight_matrix"] = shared_weight_matrix
    parameter_dict["weight_coefficients"] = weight_coefficients
    return parameter_dict

def make_coordinate_to_weights_dict(config, shared_weight_matrix, N):
    coordinate_to_weights_dict = dict()
    for i in range(0, config.dimension):
        # here we're iterating over dimension
        weight_column_vector = torch.zeros(N) # N is used here since c tensor has a thing for each node
        for k in range(0, config.dimension):
            for j in range(int(k*(N/config.dimension)), int((k+1)*(N/config.dimension))): # this is splitting up the c tensor into N/d parts
                weight_column_vector[j] += shared_weight_matrix[k][i] # we put the same number in each part
        coordinate_to_weights_dict[i] = weight_column_vector
    return coordinate_to_weights_dict

def save_model(model, example_index, config):
    if not os.path.exists(config.models_directory):
        os.makedirs(config.models_directory)
    torch.save(model.state_dict(), f'{config.models_directory}/{example_index}-model.pth')
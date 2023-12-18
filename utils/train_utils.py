import torch
from torch import nn
from .data_utils import data_set_up

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
        
        # d is the dimension of the input
        self.d = config.dimension
        num_labels = config.num_labels
        data_bounds_list = config.data_bounds
        # N is the width of the first hidden layer
        self.N = N
        self.batch_size = batch_size

        # shared_weight_matrix is a d x d matrix with rows made up of the weights that are each shared up to a constant multiple (weight_cofficients) among a set of N//d nodes of the hidden layer
        # shared_weight_matrix is an initialized as a d x d identity matrix
        self.shared_weight_matrix =  nn.Parameter(torch.eye(self.d), requires_grad=True)
        # register shared_weight_matrix so that it can be accessed using this name at any point
        self.register_parameter('shared_weight_matrix', self.shared_weight_matrix)

        # weight coefficients are the coefficients that relate the hidden layer weights to the shared_weight_matrix
        # there is one coefficient for each node in the hidden layer
        self.weight_coefficients = nn.Parameter(torch.ones(self.N), requires_grad = False)
        #self.weight_coefficients = nn.Parameter(torch.rand(self.N) + 1, requires_grad = True)
        self.register_parameter('weight_coefficients', self.weight_coefficients)

        # assign each node in the hidden layer a key between 0 and d - 1, with N // d nodes being assigned each key
        hidden_node_keys = torch.zeros(N, dtype=torch.long)
        group_size = N // self.d
        for i in range(self.d):
            hidden_node_keys[i*group_size:(i+1)*group_size] = i

        # initialize the biases (up to a constant multiple) of the hidden nodes of key = i via a uniform distribution over the interval that makes up the data domain in i-th dimension
        bias_initial_lower_bounds = torch.zeros(N)
        bias_initial_upper_bounds = torch.zeros(N)
        for i in range(self.d):
            bias_initial_lower_bounds[hidden_node_keys == i] = torch.tensor(data_bounds_list[i*2], dtype=torch.float)
            bias_initial_upper_bounds[hidden_node_keys == i] = torch.tensor(data_bounds_list[i*2 + 1], dtype=torch.float)
        
        # we multiply self.bias by self.weight_coefficient component-wise
        self.bias = nn.Parameter(torch.rand(self.N)*(bias_initial_lower_bounds - bias_initial_upper_bounds) - bias_initial_lower_bounds)
        #print('self.bias', self.bias)
        self.register_parameter('bias', self.bias)

        # define and initialize the unconstrained weights of the neural network, namely the output layers
        self.layer2 = nn.Linear(self.N, num_labels, bias = True)
        self.output_layer = nn.Linear(num_labels, 1, bias = True)

        # define the activation function of the network
        self.hardtanh1 = nn.Hardtanh(min_val = 0, max_val = 1)

        self.flatten = nn.Flatten()

    def forward(self, x):
        # create a dictionary that assigns to each dimension i, the column vector of the i-th component of each data point in the batch
        batch_component_dict = dict()
        for j in range(0, self.d):
            # select the j-th component of every data point in the batch and reshape into a column vector
            batch_component_dict[j] = torch.reshape(x[:, j], (self.batch_size, 1))

        layer1 = torch.ones(1, self.N)

        # interate over dimension
        for i in range(0, self.d):
            weights_for_dim_eq_i = torch.zeros(self.N)
            for k in range(0, self.d):
                for hidden_node in range(int(k*(self.N/self.d)), int((k+1)*(self.N/self.d))):

                    # the value of weights_for_dim_eq_i is the same up to a constant multiple for nodes of the same node key
                    weights_for_dim_eq_i[hidden_node] += (self.shared_weight_matrix[k][i] * self.weight_coefficients[hidden_node])

            # z_i keeps track of the value of the hidden nodes of key i for all of the data in the batch
            z_i = (batch_component_dict[i] * layer1) * weights_for_dim_eq_i

            # hidden_layer_before_hardtanh is a a tensor of shape batch_size x N that gets filled with the value of the hidden nodes for all of the data in the batch
            if i == 0:
                hidden_layer_before_hardtanh = z_i
            else: 
                hidden_layer_before_hardtanh += z_i

        # the operation hidden_layer_before_hardtanh + self.bias adds the N-th entry of self.bias to each entry of the N-th column of hidden_layer_before_hardtanh
        layer1_output = self.hardtanh1(hidden_layer_before_hardtanh + self.bias)

        # apply second and final layer weights 
        output = self.output_layer(self.layer2(layer1_output))
        return output
    
def get_batch_size(train_data, percentage):
    return int(len(train_data)*percentage)

''' Train loop returns the loss on the last batch of the epoch '''
def train_loop(dataloader, model, loss_fn, optimizer, batch_size, num_labels):
    num_batches = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)

        '''
        By default, .backward() in PyTorch 2.1.1 computes the accumulated gradients (see https://pytorch.org/docs/stable/_modules/torch/autograd.html)
        The following .zero_grad() call sets the accumulated gradient to zero
        '''
        
        optimizer.zero_grad()

        pred = torch.reshape(pred, (batch_size, -1))
        pred_thresh = torch.clamp(pred, min=0.0, max=float(num_labels - 1))

        loss = loss_fn(pred_thresh, y)
        loss.backward(retain_graph = True)

        optimizer.step()

        if batch == num_batches - 1:
            loss = loss.item()
            return loss
    
def test_loop(dataloader, model, loss_fn, num_labels):
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            pred_thresh = torch.clamp(pred, min=0.0, max = float(num_labels - 1))
            test_loss += loss_fn(pred_thresh, y).item()
    test_loss /= num_batches
    return test_loss
    
def train_and_test(config, N, train_data, train_dataloader, test_dataloader, batch_size):

    ''' Set up neural network and training variables'''
    loss_fn = nn.MSELoss()
    model = Regression_Cubical_Network_One_Nonlinearity(N, batch_size, config)
    optimizer = get_optimizer(config, model)

    test_loss_list = []
    train_loss_list = []

    ''' Train and test the network '''

    for epoch_number in range(config.epochs):

        # Train
        loss_train = train_loop(train_dataloader, model, loss_fn, optimizer, batch_size, config.num_labels)
        train_loss_list.append(loss_train)

        # Test
        loss_test = test_loop(test_dataloader, model, loss_fn, config.num_labels)
        test_loss_list.append(loss_test)

        if config.verbose:
            print(f"Epoch {epoch_number + 1}/{config.epochs}")
            print(f"Test loss: {loss_test:>7f}")
            print(f"Train loss: {loss_train:>7f}")

    return model, train_loss_list, test_loss_list

def save_model(model, job_index, config):
    torch.save(model.state_dict(), f'output/models/system{config.system}/jobindex{job_index}.pth')

# to do: is batch_size = None okay as the default?
def load_model(N, config, job_index, batch_size):
    cube_reg_model = Regression_Cubical_Network_One_Nonlinearity(N, batch_size, config)
    cube_reg_model.load_state_dict(torch.load(f'output/models/system{config.system}/jobindex{job_index}.pth'))
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
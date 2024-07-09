from .homology import get_homology_dict_from_model
from .train import train_and_test, compute_accuracy
from .network import load_model, get_batch_size, Regression_Cubical_Network_One_Nonlinearity
from .figure import make_decomposition_figure, plot_polytopes, make_loss_plots
from .data import data_set_up
from .decomposition import get_decomposition_data
from .config import configure
import torch

def save_model(model, file_name):
    torch.save(model.state_dict(), f'{file_name}.pth')

def load_model(system, N, file_name):
    config_fname = f'config/{system}.txt'
    config = configure(config_fname)
    cube_reg_model = Regression_Cubical_Network_One_Nonlinearity(N, 1, config)
    cube_reg_model.load_state_dict(torch.load(f'{file_name}.pth'))
    return cube_reg_model

def train_classifier(system, N, epochs, file_name):
    config_fname = f'config/{system}.txt'
    config = configure(config_fname)
    using_pandas = config.using_pandas
    train_data, test_data, train_dataloader, test_dataloader, figure_dataloader = data_set_up(config, using_pandas)
  #  if len(train_data)%10 == 0:
    batch_size = config.batch_size
    patience = config.patience
    reduction_thresh = 0.1
  #  else:
   #     batch_size = 1000 #get_batch_size(train_data, percentage = 0.1)
    trained_network, train_loss_list, test_loss_list, restart_count = train_and_test(config, N, train_dataloader, test_dataloader, batch_size, epochs, patience)
    save_model(trained_network, file_name)
    model = load_model(system, N, file_name)
    return model, train_loss_list, test_loss_list

def compute_homology(system, labeling_threshold, N, model):
    config_fname = f'config/{system}.txt'
    config = configure(config_fname)
    using_pandas = config.using_pandas
    train_data, test_data, train_dataloader, test_dataloader, figure_dataloader = data_set_up(config, using_pandas)

    sorted_hyperplane_dict, list_of_hyperplane_lists, total_hyperplane_list = get_decomposition_data(config, N, train_data, model)
        
    homology_dict, num_cubes_labeled, total_hyperplane_list, cube_list_for_polytope_figure = get_homology_dict_from_model(config, model, labeling_threshold, sorted_hyperplane_dict, list_of_hyperplane_lists, total_hyperplane_list)
    
    for label in range(config.num_labels):
        if homology_dict[label] is None:
            print('Label ' + str(label) + ' region is empty.')
        else:
            print('Betti numbers of label ' + str(label) + ' region: ' + str(homology_dict[label]))

    label = config.num_labels
    if homology_dict[label] is None:
        print('Uncertain region is empty.')
    else:
        print('Betti numbers of uncertain region: ' + str(homology_dict[label]))

    print('Number of cubes labeled: ', num_cubes_labeled)

    return total_hyperplane_list, cube_list_for_polytope_figure

def make_decomposition_plot(system, hyperplane_list, model, file_name):
    config_fname = f'config/{system}.txt'
    config = configure(config_fname)
    if config.dimension != 2:
        return 'The system has dimension greater than 2, so a plot was not produced.'
    make_decomposition_figure(config, model, hyperplane_list, True, file_name, system)

def make_polytope_plot(system, cube_list, file_name):
    config_fname = f'config/{system}.txt'
    config = configure(config_fname)
    if config.dimension != 2:
        return 'The system has dimension greater than 2, so a plot was not produced.'
    plot_polytopes(config, cube_list, True, file_name, system)

def plot_loss(system, test_loss_list, train_loss_list, file_name):
    config_fname = f'config/{system}.txt'
    config = configure(config_fname)
    fname = file_name + '.png'
    make_loss_plots(config, test_loss_list, train_loss_list, fname, True)

def accuracy(system, model, labeling_threshold):
    config_fname = f'config/{system}.txt'
    config = configure(config_fname)
    using_pandas = config.using_pandas
    train_data, test_data, train_dataloader, test_dataloader, figure_dataloader = data_set_up(config, using_pandas)
    accuracy = compute_accuracy(model, figure_dataloader, config, labeling_threshold)
    print('Accuracy using labeling threshold on test dataset: ', accuracy)
    return accuracy
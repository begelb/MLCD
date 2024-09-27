import csv
from .homology import get_homology_dict_from_model
from .train import train_and_test, compute_accuracy
from .network import save_model, load_model
from .utils import get_list_to_write
from .figure import make_decomposition_figure, make_loss_plots, plot_polytopes, plot_data, plot_multicolor_polytopes
from .data import data_set_up
from .decomposition import get_decomposition_data
from .config import user_warning_about_N_and_dimension, configure
import os
from .network import Regression_Cubical_Network_One_Nonlinearity
import torch

def change_figure(system, N, labeling_threshold_list, example_index, decomposition, polytopes, data, multicolor, model_file_name):
    config_fname = f'config/{system}.txt'
    config = configure(config_fname)

    using_pandas = config.using_pandas
    train_data, test_data, train_dataloader, test_dataloader, figure_dataloader = data_set_up(config, using_pandas = using_pandas)
    batch_size = config.batch_size

    model = Regression_Cubical_Network_One_Nonlinearity(N, 1, config)
    model.load_state_dict(torch.load(model_file_name))

    sorted_hyperplane_dict, list_of_hyperplane_lists, total_hyperplane_list = get_decomposition_data(config, N, train_data, model)
    
    for labeling_threshold in labeling_threshold_list:
        homology_dict, num_cubes_labeled, total_hyperplane_list, cube_list_for_polytope_figure = get_homology_dict_from_model(config, model, labeling_threshold, sorted_hyperplane_dict, list_of_hyperplane_lists, total_hyperplane_list)
        accuracy = compute_accuracy(model, figure_dataloader, config, labeling_threshold)
        
    print('Number of labeled cubes: ', num_cubes_labeled)

    # Make figure directory if it does not exist
    figures_directory = config.figures_directory
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory)

    if decomposition:
        decomposition_file_name = f'paper_figures_with_models/{system}/{example_index}-decomposition'
        make_decomposition_figure(config, model, total_hyperplane_list, False, decomposition_file_name, system)

    if polytopes:
        polytopes_file_name = f'paper_figures_with_models/{system}/{example_index}-polytopes'
        plot_polytopes(config, cube_list_for_polytope_figure, False, polytopes_file_name, system)
    
    if data:
        data_file_name = f'paper_figures_with_models/{system}/{example_index}-data'
        plot_data(system, config, data_file_name)

    if multicolor:
        multicolor_file_name = f'paper_figures_with_models/{system}/{example_index}-multicolor'
        plot_multicolor_polytopes(config, cube_list_for_polytope_figure, False, multicolor_file_name, system)
        

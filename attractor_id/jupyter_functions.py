from .homology import get_homology_dict_from_model
from .train import train_and_test
from .network import save_model, load_model, get_batch_size
from .figure import make_figure, make_loss_plots
from .data import data_set_up
from .decomposition import get_decomposition_data
from .config import configure
#from .utils import system_name_to_number

def train_classifier(system_name, N, epochs, example_index=0):
    system = system_name_to_number(system_name)
    config_fname = f'config/system{system}.txt'
    config = configure(config_fname)
    using_pandas = config.using_pandas
    train_data, test_data, train_dataloader, test_dataloader, figure_dataloader = data_set_up(config, using_pandas)
    batch_size = get_batch_size(train_data, percentage = 0.1)
    trained_network, train_loss_list, test_loss_list = train_and_test(config, N, train_dataloader, test_dataloader, batch_size, epochs)
    save_model(trained_network, example_index, config)
    model = load_model(N, config, 1, example_index)
    return model

def compute_homology(system_name, labeling_threshold, N, model):
    system = system_name_to_number(system_name)
    config_fname = f'config/system{system}.txt'
    config = configure(config_fname)
    using_pandas = config.using_pandas
    train_data, test_data, train_dataloader, test_dataloader, figure_dataloader = data_set_up(config, using_pandas)

    sorted_hyperplane_dict, list_of_hyperplane_lists, total_hyperplane_list = get_decomposition_data(config, N, train_data)
        
    homology_dict, num_cubes_labeled, total_hyperplane_list = get_homology_dict_from_model(config, model, labeling_threshold, sorted_hyperplane_dict, list_of_hyperplane_lists, total_hyperplane_list)
    
    for label in range(config.num_labels + 1):
        if homology_dict[label] is None:
            print('Label ' + str(label) + ' region is empty.')
        else:
            print('Betti numbers of label ' + str(label) + ' region: ' + str(homology_dict[label]))

    print('Number of cubes labeled: ', num_cubes_labeled)

    return total_hyperplane_list

def make_decomposition_plot(system_name, N, hyperplane_list, example_index=0):
    system = system_name_to_number(system_name)
    config_fname = f'config/system{system}.txt'
    config = configure(config_fname)
    if config.dimension != 2:
        return 'The system has dimension greater than 2, so a plot was not produced.'
    using_pandas = config.using_pandas
    train_data, test_data, train_dataloader, test_dataloader, figure_dataloader = data_set_up(config, using_pandas)
    model = load_model(N, config, 1, example_index=0)

    make_figure(config, figure_dataloader, model, test_data, hyperplane_list, show = True)
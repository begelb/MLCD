import csv
from .homology import get_homology_dict_from_model
from .train import train_and_test, compute_accuracy
from .network import save_model, load_model
from .utils import get_list_to_write
from .figure import make_decomposition_figure, make_loss_plots, plot_polytopes
from .data import data_set_up
from .decomposition import get_decomposition_data
from .config import user_warning_about_N_and_dimension, configure
import os

def compute_example(system, N, labeling_threshold_list, example_index=0):
    config_fname = f'config/{system}.txt'
    config = configure(config_fname)
    user_warning_about_N_and_dimension(config, N)

    # Make results directory if it does not exist
    results_directory = config.results_directory
    if not os.path.isdir(results_directory):
        os.makedirs(results_directory)

    with open(f'{results_directory}/{example_index}-results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ex_num", "N", "optimizer_choice", "learning_rate", "epsilon", "num_cubes", "final_test_loss", "test_accuracy", "hom_uncertain", "hom_zero", "hom_one", "hom_two", "hom_three"])

        using_pandas = config.using_pandas
        train_data, test_data, train_dataloader, test_dataloader, figure_dataloader = data_set_up(config, using_pandas = using_pandas)
        batch_size = config.batch_size
        epochs = config.epochs
        trained_network, train_loss_list, test_loss_list = train_and_test(config, N, train_dataloader, test_dataloader, batch_size, epochs)
        save_model(trained_network, example_index, config)
        model = load_model(N, system, config, 1, example_index)
        sorted_hyperplane_dict, list_of_hyperplane_lists, total_hyperplane_list = get_decomposition_data(config, N, train_data, model)
        
        for labeling_threshold in labeling_threshold_list:
            homology_dict, num_cubes_labeled, total_hyperplane_list, cube_list_for_polytope_figure = get_homology_dict_from_model(config, model, labeling_threshold, sorted_hyperplane_dict, list_of_hyperplane_lists, total_hyperplane_list)
            accuracy = compute_accuracy(model, figure_dataloader, config, labeling_threshold)
            list_to_write = get_list_to_write(config, example_index, N, labeling_threshold, num_cubes_labeled, test_loss_list, homology_dict, accuracy)
            writer.writerow(list_to_write)
            file.flush()

    if config.make_figures:

        # Make figure directory if it does not exist
        figures_directory = config.figures_directory
        if not os.path.isdir(figures_directory):
            os.makedirs(figures_directory)
        
        loss_plot_file_name = f'output/figures/{system}/{example_index}-loss.png'

        make_loss_plots(config, system, test_loss_list, train_loss_list, loss_plot_file_name, False)

        if config.dimension == 2:
            model = load_model(N, system, config, 1, example_index)
            decomposition_file_name = f'{figures_directory}/{example_index}-decomposition.png'
            make_decomposition_figure(config, model, total_hyperplane_list, False, decomposition_file_name)

            polytopes_file_name = f'{figures_directory}/{example_index}-polytopes.png'
            plot_polytopes(config, cube_list_for_polytope_figure, False, polytopes_file_name)


        

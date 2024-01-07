import csv
from .homology import get_homology_dict_from_model
from .train import train_and_test
from .network import save_model, load_model, get_batch_size
from .misc import get_list_to_write
from .figure import make_figure, make_loss_plots
from .data import data_set_up
from .decomposition import get_decomposition_data

def compute_example(config, example_index, N, labeling_threshold_list):
    with open(f'output/results/system{config.system}/{example_index}-results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ex_num", "N", "optimizer_choice", "learning_rate", "epsilon", "num_cubes", "final_test_loss", "hom_uncertain", "hom_zero", "hom_one", "hom_two", "hom_three"])

        train_data, train_dataloader, test_dataloader, figure_dataloader = data_set_up(config)
        batch_size = get_batch_size(train_data, percentage = 0.1)
        trained_network, train_loss_list, test_loss_list = train_and_test(config, N, train_dataloader, test_dataloader, batch_size)
        save_model(trained_network, example_index, config)
        model = load_model(N, config, example_index, batch_size = 1)
        sorted_hyperplane_dict, list_of_hyperplane_lists, total_hyperplane_list = get_decomposition_data(config, N, example_index, batch_size, train_data)
        
        for labeling_threshold in labeling_threshold_list:
            homology_dict, num_cubes_labeled, total_hyperplane_list = get_homology_dict_from_model(config, model, labeling_threshold, sorted_hyperplane_dict, list_of_hyperplane_lists, total_hyperplane_list)
            list_to_write = get_list_to_write(config, example_index, N, labeling_threshold, num_cubes_labeled, test_loss_list, homology_dict)
            writer.writerow(list_to_write)
            file.flush()

        if config.make_figures:
            model = load_model(N, config, example_index, batch_size = batch_size)
            make_figure(config, figure_dataloader, model, train_data, total_hyperplane_list, example_index)
            make_loss_plots(config, example_index, test_loss_list, train_loss_list)

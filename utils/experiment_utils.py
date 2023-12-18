import csv
from utils.decomposition_utils import get_homology_dict_from_model
from utils.config_utils import configure
from utils.train_utils import train_and_test, save_model, load_model, get_batch_size
from utils.misc_utils import get_list_to_write
from utils.figure_utils import make_figure, make_loss_plots
from utils.data_utils import data_set_up

def compute_example(config, job_index, N, labeling_threshold):
    with open(f'output/results/system{config.system}/results_system{config.system}_example_{job_index}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ex_num", "N", "optimizer_choice", "learning_rate", "epsilon", "num_cubes", "final_test_loss", "hom_uncertain", "hom_zero", "hom_one", "hom_two", "hom_three"])

        train_data, train_dataloader, test_dataloader, figure_dataloader = data_set_up(config)
        batch_size = get_batch_size(train_data, percentage = 0.1)
        trained_network, train_loss_list, test_loss_list = train_and_test(config, N, train_data, train_dataloader, test_dataloader, batch_size)
        save_model(trained_network, job_index, config)
        model = load_model(N, config, job_index, batch_size = 1)
        homology_dict, num_cubes_labeled, total_hyperplane_list = get_homology_dict_from_model(config, N, model, train_data, job_index, labeling_threshold, batch_size = 1)
        list_to_write = get_list_to_write(config, job_index, N, labeling_threshold, num_cubes_labeled, test_loss_list, homology_dict)
        writer.writerow(list_to_write)
        file.flush()

        if config.make_figures:
            model = load_model(N, config, job_index, batch_size = batch_size)
            make_figure(config, figure_dataloader, model, train_data, total_hyperplane_list, job_index)
            make_loss_plots(config, job_index, test_loss_list, train_loss_list)


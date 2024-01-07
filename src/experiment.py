import csv
from .homology import get_homology_dict_from_model
from .train import train_and_test
from .network import save_model, load_model, get_batch_size
from .misc import get_list_to_write
from .figure import make_figure, make_loss_plots
from .data import data_set_up
from .compute_example import compute_example

class Experiment:
    def __init__(self, labeling_threshold_list, N_list):
        self.labeling_threshold_list = labeling_threshold_list
        self.N_list = N_list

    def generate_parameter_lists(self):
        parameter_list = []
        for N in self.N_list:
            for labeling_threshold in self.labeling_threshold_list:
                parameter_list.append([N, labeling_threshold])
        return parameter_list

    def get_experiment_parameters(self, example_index):
        return self.generate_parameter_lists()[example_index]
    
    def run_experiment(self, job_index):
        parameter_list = self.generate_parameter_lists()
        num_parameter_combinations = len(parameter_list)
        for i in range(num_parameter_combinations):
            example_index = job_index * num_parameter_combinations + i
            N = parameter_list[i][0]
            labeling_threshold = parameter_list[i][1]
            compute_example(config, example_index, N, labeling_threshold)

def compute_example(config, example_index, N, labeling_threshold):
    with open(f'output/results/system{config.system}/{example_index}-results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ex_num", "N", "optimizer_choice", "learning_rate", "epsilon", "num_cubes", "final_test_loss", "hom_uncertain", "hom_zero", "hom_one", "hom_two", "hom_three"])

        train_data, train_dataloader, test_dataloader, figure_dataloader = data_set_up(config)
        batch_size = get_batch_size(train_data, percentage = 0.1)
        trained_network, train_loss_list, test_loss_list = train_and_test(config, N, train_dataloader, test_dataloader, batch_size)
        save_model(trained_network, example_index, config)
        model = load_model(N, config, example_index, batch_size = 1)
        homology_dict, num_cubes_labeled, total_hyperplane_list = get_homology_dict_from_model(config, N, model, train_data, example_index, labeling_threshold, batch_size = 1)
        list_to_write = get_list_to_write(config, example_index, N, labeling_threshold, num_cubes_labeled, test_loss_list, homology_dict)
        writer.writerow(list_to_write)
        file.flush()

        if config.make_figures:
            model = load_model(N, config, example_index, batch_size = batch_size)
            make_figure(config, figure_dataloader, model, train_data, total_hyperplane_list, example_index)
            make_loss_plots(config, example_index, test_loss_list, train_loss_list)


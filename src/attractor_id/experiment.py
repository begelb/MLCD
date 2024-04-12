from .compute_example import compute_example

class Experiment:
    def __init__(self, N_list):
        self.N_list = N_list

    def generate_parameter_lists(self):
        parameter_list = []
        for N in self.N_list:
            parameter_list.append([N])
        return parameter_list

    def get_experiment_parameters(self, parameter_list, example_index):
        return parameter_list[example_index]
    
    # num of jobs run should be equal to the number of repetitions per parameter set 
    def run_experiment(self, job_index, system, repetitions_per_parameter_set, labeling_threshold_list):
        parameter_list = self.generate_parameter_lists()
        for param_index in range(len(parameter_list)):
            example_index = param_index * repetitions_per_parameter_set + job_index
            N = parameter_list[param_index][0]
            compute_example(system, N, labeling_threshold_list, example_index)

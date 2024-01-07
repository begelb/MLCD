
repetitions_per_parameter_set = 100
job_index = 4

def run_experiment(job_index, repetitions_per_parameter_set):
    parameter_list = [0, 1, 2, 3, 4, 6, 7, 8]
    for param_index in range(len(parameter_list)):
        example_index = param_index * repetitions_per_parameter_set + job_index
        print('example_index: ', example_index)

run_experiment(job_index, repetitions_per_parameter_set)
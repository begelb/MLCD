def get_initial_list_to_write(config, job_index, N, labeling_threshold, num_cubes_labeled, test_loss_list):
    initial_list = [job_index, N, config.optimizer_choice, config.learning_rate, labeling_threshold, num_cubes_labeled, test_loss_list[-1],]
    return initial_list

def get_final_list_to_write(config, initial_list, homology_dict):
    for label in range(config.num_labels + 1):
        initial_list.append(homology_dict[label])
    return initial_list

def get_list_to_write(config, job_index, N, labeling_threshold, num_cubes_labeled, test_loss_list, homology_dict):
    initial_list = get_initial_list_to_write(config, job_index, N, labeling_threshold, num_cubes_labeled, test_loss_list)
    return get_final_list_to_write(config, initial_list, homology_dict)
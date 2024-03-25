def get_initial_list_to_write(config, example_index, N, labeling_threshold, num_cubes_labeled, test_loss_list):
    initial_list = [example_index, N, config.optimizer_choice, config.learning_rate, labeling_threshold, num_cubes_labeled, test_loss_list[-1],]
    return initial_list

def get_final_list_to_write(config, initial_list, homology_dict):

    # first append the homology of the uncertain region
    if homology_dict[config.num_labels] is not None:
        initial_list.append(homology_dict[config.num_labels])
    else:
        initial_list.append("Empty region")

    # then append the homology of the remaining regions
    for label in range(config.num_labels):
        if homology_dict[label] is not None:
            initial_list.append(homology_dict[label])
        else:
            initial_list.append("Empty region")
    return initial_list

def get_list_to_write(config, example_index, N, labeling_threshold, num_cubes_labeled, test_loss_list, homology_dict):
    initial_list = get_initial_list_to_write(config, example_index, N, labeling_threshold, num_cubes_labeled, test_loss_list)
    return get_final_list_to_write(config, initial_list, homology_dict)
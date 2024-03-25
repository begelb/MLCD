def get_initial_list_to_write(config, example_index, N, labeling_threshold, num_cubes_labeled, test_loss_list):
    initial_list = [example_index, N, config.optimizer_choice, config.learning_rate, labeling_threshold, num_cubes_labeled, test_loss_list[-1],]
    return initial_list

def get_final_list_to_write(config, initial_list, homology_dict):
    for label in range(config.num_labels + 1):
        if homology_dict[label] is not None:
            initial_list.append(homology_dict[label])
        else:
            initial_list.append("Empty region")
    return initial_list

def get_list_to_write(config, example_index, N, labeling_threshold, num_cubes_labeled, test_loss_list, homology_dict):
    initial_list = get_initial_list_to_write(config, example_index, N, labeling_threshold, num_cubes_labeled, test_loss_list)
    return get_final_list_to_write(config, initial_list, homology_dict)

def system_name_to_number(name):
    if name == 'straight_separatrix':
        return 1
    if name == 'radial_2_label':
        return 2
    if name == 'radial_3_label':
        return 3
    if name == 'curved_separatrix':
        return 4
    if name == 'EMT':
        return 5
    if name == 'periodic':
        return 6
    if name == 'ellipsoidal_2d':
        return 7
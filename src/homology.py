from .label_cubes import get_labeled_cubes, get_num_cubes_labeled
import pychomp2 as pychomp

def get_homology_dict_from_model(config, model, labeling_threshold, sorted_hyperplane_dict, list_of_hyperplane_lists, total_hyperplane_list):
    label_to_cubes_dict = get_labeled_cubes(config, sorted_hyperplane_dict, list_of_hyperplane_lists, model, labeling_threshold)
    num_cubes_labeled = get_num_cubes_labeled(config, label_to_cubes_dict)
    homology_dict = get_label_to_homology_dict(config, label_to_cubes_dict)

    return homology_dict, num_cubes_labeled, total_hyperplane_list

def get_homology(config, label_to_cubes_dict):
    for label in range(config.num_labels):
        cube_list = label_to_cubes_dict[label]
        if len(cube_list) == 0:
            homology = []
        else:
            homology = pychomp.CubicalHomology(cube_list)
    return homology

def get_label_to_homology_dict(config, label_to_cubes_dict):
    label_to_homology_dict = dict()
    for label in range(config.num_labels + 1):
        label_to_homology_dict[label] = get_homology(config, label_to_cubes_dict)
    return label_to_homology_dict
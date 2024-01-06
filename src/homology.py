from .decomposition import make_hyperplane_dicts, get_hyperplane_data
from .label_cubes import get_labeled_cubes, get_num_cubes_labeled
from .network import load_model, get_model_parameters, make_coordinate_to_weights_dict
from .data import convert_data_to_tensors
import pychomp2 as pychomp

def get_homology_dict_from_model(config, N, model, data, job_index, labeling_threshold, batch_size = 1):
    cube_reg_model = load_model(N, config, job_index, batch_size)
    parameter_dict = get_model_parameters(cube_reg_model)

    coordinate_to_weights_dict = make_coordinate_to_weights_dict(config, parameter_dict["shared_weight_matrix"], N)

    data_as_tensors = convert_data_to_tensors(data, config.dimension)
    hyperplane_dict, is_boundary_hyperplane_dict = make_hyperplane_dicts(config, coordinate_to_weights_dict, N, parameter_dict["biaslist"], data_as_tensors, parameter_dict["weight_coefficients"])
    list_of_hyperplane_lists, sorted_hyperplane_dict, total_hyperplane_list = get_hyperplane_data(config, hyperplane_dict, data_as_tensors, is_boundary_hyperplane_dict)

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
import numpy as np
from .hyperplane import Hyperplane, data_is_on_one_side_of_hyperplane
from copy import deepcopy
from .network import get_model_parameters, make_coordinate_to_weights_dict
from .data import convert_data_to_tensors

def add_boundaries(config, hyperplane_list, data_as_tensors, is_boundary_hyperplane_dict):
    d = config.dimension
    data_bounds_list = config.data_bounds
    normal = hyperplane_list[0].normal_vec

    # Initialize two hyperplanes that are parallel to the hyperplanes in hyperplane list
    lower_bound = data_bounds_list[0][0]
    upper_bound = data_bounds_list[0][1]
    # The arguments lower_bound - 1 and upper_boundary + 1 are relatively arbitrary here
    # It is not a mistake to use the bounds of the first dimension for all dimensions
    # The goal is to start not too close to the origin so that move_outer_boundary finishes quickly
    neg_boundary = Hyperplane(normal, lower_bound - 1)
    pos_boundary = Hyperplane(normal, upper_bound + 1)

    # Move the hyperplanes until they are outside the domain
    new_pos_hyperplane = move_outer_boundary(neg_boundary, data_as_tensors)
    new_neg_hyperplane = move_outer_boundary(pos_boundary, data_as_tensors)

    # Add the hyperplanes to hyperplane list and is_boundary_hyperplane_dict
    hyperplane_list.append(new_pos_hyperplane)
    hyperplane_list.append(new_neg_hyperplane)
    is_boundary_hyperplane_dict[new_neg_hyperplane] = True
    is_boundary_hyperplane_dict[new_pos_hyperplane] = True

    # Return updated hyperplane_list and updated is_boundary_hyperplane_dict
    return hyperplane_list, is_boundary_hyperplane_dict

def move_outer_boundary(hyperplane, data_as_tensors):
    KeepMoving = True
    while KeepMoving:
        hyperplane = hyperplane.move_away_from_origin(percentage = 10)
        if data_is_on_one_side_of_hyperplane(hyperplane, data_as_tensors):
            KeepMoving = False
    return hyperplane

def sort_hyperplanes_and_remove_unnecessary_hyperplanes(hyperplane_list, is_boundary_hyperplane_dict):

    # Sort the hyperplanes according to their biases in increasing order
    sorted_list = sort_hyperplanes_by_bias(hyperplane_list)
    refined_list = []

    # Iterate through the sorted hyperplanes; once you get to a hyperplane such that all of the data is NOT on one side of it, add the preceeding hyperplane to a new, refined list; then stop iterating

    for i, hyperplane in enumerate(sorted_list):
        if is_boundary_hyperplane_dict[hyperplane] == False:
            refined_list.append(sorted_list[i-1])
            break
    

    # Iterate through the sorted hyperplanes in reverse order; once you get to a hyperplane such that all of the data is NOT on one side of it, add the preceeding hyperplane to the new, refined list; then stop iterating
    sorted_list.reverse()
    for i, hyperplane in enumerate(sorted_list):
        if is_boundary_hyperplane_dict[hyperplane] == False:
            refined_list.append(sorted_list[i-1])
            break
    
    # Add all hyperplanes such that all of the data is NOT on one side of it to the new list
    for hyperplane in sorted_list:
        if is_boundary_hyperplane_dict[hyperplane] == False:
            refined_list.append(hyperplane)

    if len(refined_list) == 0: # This covers the case when all hyperplanes are boundary hyperplanes
        refined_list.append(sorted_list[0])
        refined_list.append(sorted_list[-1])
        #raise Exception('The learned cubical complex does not intersect the domain. Program terminated.')

    # Resort the new list of hyperplanes by bias
    final_list = sort_hyperplanes_by_bias(refined_list)

    return final_list

def next_cube_index(cube_index, d, num_of_hyperplanes_dict):
    new_cube_index = deepcopy(cube_index)
    new_cube_index[-1]+=1
    for i in reversed(range(0, d)):
        if new_cube_index[i] == num_of_hyperplanes_dict[i] - 1:
            new_cube_index[i] = 0
            new_cube_index[i-1] += 1
    return new_cube_index

def sort_hyperplanes_by_bias(hyperplane_list):
    hyperplane_list.sort(key = lambda x: x.bias)
    return hyperplane_list

def get_group_index_to_num_hyperplanes_dict(list_of_hyperplane_lists):
    num_of_hyperplanes_dict = dict()
    for j in range(0, len(list_of_hyperplane_lists)):
        num_of_hyperplanes_dict[j] = len(list_of_hyperplane_lists[j])
    return num_of_hyperplanes_dict

def make_hyperplane_dicts(config, c_tensor_dict, N, biaslist, data_as_tensors, weight_coefficients):
    d = config.dimension

    hyperplane_dict = dict()
    is_boundary_hyperplane_dict = dict()
    total_hyperplane_list = []
    for k in range(0, d):
        #print('making hyperplane dict dim: ', k)
        hyperplane_list = []
        for j in range(int(k*(N/d)), int((k+1)*(N/d))):
            normal_vector = np.ones(d)
            for m in range(0, d): # m is for the variable dimension
                c_value = c_tensor_dict[m][j] # j is for the node
                normal_vector[m] = c_value
            
            normalizing_factor = 1 / weight_coefficients[j]
            bias = biaslist[j]

            new_hyperplane_list = []

            H1 = Hyperplane(normal_vector, bias)
            new_hyperplane_list.append(H1)
            H2 = Hyperplane(normal_vector, bias - normalizing_factor)
            new_hyperplane_list.append(H2)
            
            total_hyperplane_list.extend(new_hyperplane_list)
            hyperplane_list.extend(new_hyperplane_list)
            for H in new_hyperplane_list:
                if data_is_on_one_side_of_hyperplane(H, data_as_tensors):
                    is_boundary_hyperplane_dict[H] = True
                else:
                    is_boundary_hyperplane_dict[H] = False
        hyperplane_dict[k] = hyperplane_list
    return hyperplane_dict, is_boundary_hyperplane_dict

def get_hyperplane_data(config, hyperplane_dict, data_as_tensors, is_boundary_hyperplane_dict):
    list_of_hyperplane_lists = []
    sorted_hyperplane_dict = dict()
    total_hyperplane_list = []

    for k in range(0, config.dimension):
        list_of_hyperplanes_with_boundaries, is_boundary_hyperplane_dict = add_boundaries(config, hyperplane_dict[k], data_as_tensors, is_boundary_hyperplane_dict)
        sorted_hyperplane_list_with_single_boundaries = sort_hyperplanes_and_remove_unnecessary_hyperplanes(list_of_hyperplanes_with_boundaries, is_boundary_hyperplane_dict)
        list_of_hyperplane_lists.append(sorted_hyperplane_list_with_single_boundaries)
        total_hyperplane_list.extend(sorted_hyperplane_list_with_single_boundaries)
        sorted_hyperplane_dict[k] = sorted_hyperplane_list_with_single_boundaries
    return list_of_hyperplane_lists, sorted_hyperplane_dict, total_hyperplane_list


def get_decomposition_data(config, N, data, model):
    parameter_dict = get_model_parameters(model)
    coordinate_to_weights_dict = make_coordinate_to_weights_dict(config, parameter_dict["shared_weight_matrix"], N)
    data_as_tensors = convert_data_to_tensors(data, config.dimension)
    hyperplane_dict, is_boundary_hyperplane_dict = make_hyperplane_dicts(config, coordinate_to_weights_dict, N, parameter_dict["biaslist"], data_as_tensors, parameter_dict["weight_coefficients"])
    list_of_hyperplane_lists, sorted_hyperplane_dict, total_hyperplane_list = get_hyperplane_data(config, hyperplane_dict, data_as_tensors, is_boundary_hyperplane_dict)

    return sorted_hyperplane_dict, list_of_hyperplane_lists, total_hyperplane_list

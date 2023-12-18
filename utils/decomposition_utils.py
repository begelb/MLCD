import numpy as np
import torch
from copy import deepcopy
import pychomp2 as pychomp
from itertools import combinations
import polytope as pc # It is also recommended to import cvxopt, which is a non-required dependency of polytope
from .train_utils import load_model, get_model_parameters, make_coordinate_to_weights_dict

torch.autograd.set_detect_anomaly(True)

# To do: Re-organize the order of functions

def generate_domain_bounding_hyperplanes(config):
    data_bounds_list = config.data_bounds
    d = len(data_bounds_list)//2
    domain_bounding_hyperplanes = []
    for i in range(0, d):
        normal_vec = np.zeros(d)
        normal_vec[i] = 1
        H1 = Hyperplane(normal_vec, -data_bounds_list[2*i]) # Why is this minus and not plus?
        H2 = Hyperplane(normal_vec, -data_bounds_list[2*i + 1])
        domain_bounding_hyperplanes.extend([H1, H2])
    return domain_bounding_hyperplanes

''' Function that returns output of regression network for a given input after applying thresholding '''
def evaluate_regression_network(config, x, model):
    d = config.dimension
    num_labels = config.num_labels
    return torch.clamp(model(x.clone().detach().view(1, d)), min = 0.0, max = float(num_labels) - 1)

''' Compute the intersection of d hyperplanes in general arrangement in dimension d '''
def hyperplane_intersection(hyperplane_list, d):
    N = torch.zeros(d, d)
    b = torch.zeros(d)
    for i in range(0, d):
        N[i] = torch.from_numpy(hyperplane_list[i].normal_vec)
        b[i] = -hyperplane_list[i].bias
    # solve and return the system of linear equations Nx = b
    return torch.linalg.solve(N, b)

''' Compute all possible binary lists of length d '''
def make_all_binary_lists(length):
    # base case for recursive definition
    if length == 1:
        return [[0], [1]]
    else:
        list_d_minus_one = make_all_binary_lists(length - 1)
        list_d_minus_one_copy = deepcopy(list_d_minus_one)

        # for each list-element of these lists, insert 0 or 1 respectively at the end of the list-element
        for i in range(0, len(list_d_minus_one)):
            list_d_minus_one[i].insert(0, 0)
            list_d_minus_one_copy[i].insert(0, 1)

        # return the concatenation of the modified lists
        return list_d_minus_one + list_d_minus_one_copy

''' Compute the vertices of a d-dimensional parallelepiped given a hyperplane list such that hyperplane_list[i] and hyperplane_list[i+1] are parallel if and only if i % 2 == 0 '''
def parallelepiped_vertices(hyperplane_list, d):
    # Construct a list of parallel hyperplane pairs
    list_of_parallel_hyperplane_pairs = []
    for i in range(0, len(hyperplane_list)):
        if i % 2 == 0:
            list_of_parallel_hyperplane_pairs.append([hyperplane_list[i], hyperplane_list[i+1]])

    # Construct list of all binary lists of length d
    binary_list = make_all_binary_lists(length = d)

    vertex_list = []
    # Use each binary list to determine which hyperplanes to compute intersection of
    for index_list in binary_list:
        hyperplanes_to_get_intersection_list = []
        for i, elem in enumerate(index_list):
            hyperplanes_to_get_intersection_list.append(list_of_parallel_hyperplane_pairs[i][elem])
        vertex_list.append(hyperplane_intersection(hyperplanes_to_get_intersection_list, d))
    return vertex_list

def get_combinations_minus_one_element(my_list):
    j = len(my_list) - 1
    return list(combinations(my_list, j))

def point_inside_domain(point, data_bounds_list, d):
    tolerance = 1e-4
    pt_inside_domain_Bool_tracker = []
    for i in range(d):
        if float(point[i]) + tolerance < data_bounds_list[2*i] or float(point[i]) - tolerance > data_bounds_list[2*i+1]:
            pt_inside_domain_Bool_tracker.append(False)
        else:
            pt_inside_domain_Bool_tracker.append(True)
    if all(pt_inside_domain_Bool_tracker):
        return True
    else:
        return False


''' Compute the vertices of the polytope that is the intersection of a cubical data domain and a d-dimensional parallelepiped given a hyperplane list such that hyperplane_list[i] and hyperplane_list[i+1] are parallel if and only if i % 2 == 0 '''

def vertices_of_parallelepiped_intersected_with_domain(hyperplane_list, d, data_bounds_list, domain_bounding_hyperplanes, cube_object, binary_lists):
    
    # Construct a list of parallel hyperplane pairs
    list_of_parallel_hyperplane_pairs = []
    for i in range(0, len(hyperplane_list)):
        if i % 2 == 0:
            list_of_parallel_hyperplane_pairs.append([hyperplane_list[i], hyperplane_list[i+1]])

    vertex_list = []

    # Use each binary list to determine which hyperplanes to compute the intersection of
    for index_list in binary_lists:
        hyperplanes_to_get_intersection_list = []
        for i, elem in enumerate(index_list):
            hyperplanes_to_get_intersection_list.append(list_of_parallel_hyperplane_pairs[i][elem])
        intersection_pt = hyperplane_intersection(hyperplanes_to_get_intersection_list, d)

        # Check if the intersection point lies inside the data domain by checking along each dimension
        # The while loop will keep going until all dimensions are checked OR until a dimension is found such that the point is outside the domain along that dimension
        pt_inside_domain_Bool_tracker = [True]

        # the variable i enumerates the dimension that is being checked
        i = 0

        while all(pt_inside_domain_Bool_tracker) and i < d:

            # Check if the the intersection point is outside the data domain along some dimension
            if float(intersection_pt[i]) < data_bounds_list[2*i] or float(intersection_pt[i]) > data_bounds_list[2*i+1]:

                # The following code is only evaluate one time
                pt_inside_domain_Bool_tracker.append(False)
                
                # Make a list that is a list of lists with len(intersection) - 1 elements from the original intersection_list
                list_of_new_intersection_lists = get_combinations_minus_one_element(hyperplanes_to_get_intersection_list)
                for combination_list in list_of_new_intersection_lists:
                    
                    # Iterate over a domain bounding hyperplane that is added to that list
                    for h in domain_bounding_hyperplanes:
                        
                        # Check if this new list has an intersection
                        try:
                            new_vertex = hyperplane_intersection(list(combination_list) + [h], d)
                            
                            # Check if the intersection is inside the parallelepiped
                            if inside_parallelepiped(new_vertex, cube_object) and point_inside_domain(new_vertex, data_bounds_list, d):
                                
                                # Append to the vertex list if inside the parallelepiped and if inside the data domain
                                vertex_list.append(new_vertex)
                                #break
                        except:
                            pass

            else:
                pt_inside_domain_Bool_tracker.append(True)

            # progress to the next dimension
            i += 1

        # Check if the point is inside the data domain along all dimensions; if so, append to the vertex list
        if all(pt_inside_domain_Bool_tracker):
            vertex_list.append(intersection_pt)

    return vertex_list

class Hyperplane:
    def __init__(self, normal_vector, bias):
        self.normal_vec = normal_vector
        self.bias = bias

    def eval_hyperplane(self, vec):
        return np.dot(np.asarray(vec), np.asarray(self.normal_vec)) + self.bias

    def unit_normal(self):
        return np.asarray(self.normal_vec, dtype = float) * float(1/np.linalg.norm(self.normal_vec))

    def move_away_from_zero(self, percentage):
        b = self.bias
        new_b = b*percentage
        return Hyperplane(self.normal_vec, new_b)

# write a function that takes away boundary hyperplanes until only the first and last hyperplanes are boundary hyperplanes
def sort_hyperplanes_and_remove_unecessary_hyperplanes(hyperplane_list, is_boundary_hyperplane_dict):

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

    # HOW TO MAKE THE CODE JUST TERMINATE AT THIS POINT?
    if len(refined_list) == 0: # This covers the case when all hyperplanes are boundary hyperplanes
        refined_list.append(sorted_list[0])
        refined_list.append(sorted_list[-1])

    # Resort the new list of hyperplanes by bias
    final_list = sort_hyperplanes_by_bias(refined_list)

    return final_list

def convert_data_to_tensors(data, d):
    data_tensor_list = []
    for l in range(0, len(data)):
        data_point = data[l]
        data_tensor = np.zeros(d)
        for k in range(0, d):
            data_tensor[k] += float(data_point[k])
        data_tensor_list.append(data_tensor)
    return data_tensor_list

def data_is_on_one_side_of_hyperplane(hyperplane, data_as_tensors, d):

    # We will keep track of the sign of the data points when evaluated at the hyperplane equation, and ultimately return True if all the signs are the same and False if at least two signs are different

    exists_positive = False
    exists_negative = False

    for data_point in data_as_tensors:
        if np.sign(hyperplane.eval_hyperplane(data_point)) == 1:
            exists_positive = True
        else:
            exists_negative = True

        # Stop evaluating data points if there are two signs that are different
        if exists_positive and exists_negative:
            break

    if exists_positive and exists_negative:
        return False
    else:
        return True

def move_outer_boundary(hyperplane, data_as_tensors, d):
    KeepMoving = True
    while KeepMoving:
        hyperplane = hyperplane.move_away_from_zero(percentage = 10)
        if data_is_on_one_side_of_hyperplane(hyperplane, data_as_tensors, d):
            KeepMoving = False
    return hyperplane

def add_boundaries(config, hyperplane_list, data_as_tensors, is_boundary_hyperplane_dict):
    d = config.dimension
    data_bounds_list = config.data_bounds
    normal = hyperplane_list[0].normal_vec
    lower_bound = data_bounds_list[d]
    upper_bound = data_bounds_list[d+1]
    neg_boundary = Hyperplane(normal, lower_bound - 1)
    pos_boundary = Hyperplane(normal, upper_bound + 1)
    new_pos_hyperplane = move_outer_boundary(neg_boundary, data_as_tensors, d)
    new_neg_hyperplane = move_outer_boundary(pos_boundary, data_as_tensors, d)
    hyperplane_list.append(new_pos_hyperplane)
    hyperplane_list.append(new_neg_hyperplane)
    is_boundary_hyperplane_dict[new_neg_hyperplane] = True
    is_boundary_hyperplane_dict[new_pos_hyperplane] = True
    return hyperplane_list, is_boundary_hyperplane_dict

def next_cube_index(cube_index, d, num_of_hyperplanes_dict):
    new_cube_index = deepcopy(cube_index)
    new_cube_index[-1]+=1
    for i in reversed(range(0, d)):
        if new_cube_index[i] == num_of_hyperplanes_dict[i] - 1:
            new_cube_index[i] = 0
            new_cube_index[i-1] += 1
    return new_cube_index

''' Check if a point is between two hyperplanes that are assumed to be parallel '''
def between_parallel_hyperplanes(point, hyperplane1, hyperplane2):
    # Tolerance is the amount of numerical error that is allowed
    tolerance = 1e-8

    # Obtain the unit normal vectors and the biases of the hyperplanes, normalized by the norm of the normal vector
    hyperplane1_unit_normal = np.asarray(hyperplane1.unit_normal(), dtype = float)
    hyperplane2_unit_normal = np.asarray(hyperplane2.unit_normal(), dtype = float)

    hyperplane1_bias = hyperplane1.bias/np.linalg.norm(hyperplane1.normal_vec)
    hyperplane2_bias = hyperplane2.bias/np.linalg.norm(hyperplane2.normal_vec)

    # By assumption, the unit normal vectors are the same up to sign, so we check if the signs are the same, and multiply the coefficients of the equation of the second hyperplane if not
    if not np.sign(hyperplane1_unit_normal[0]) == np.sign(hyperplane2_unit_normal[0]):
        hyperplane2_unit_normal = -1 * hyperplane2_unit_normal
        hyperplane2_bias = -1 * hyperplane2_bias

    # We check if the point is in between the two hyperplanes by checking if the sign of the hyperplanes evaluated at the point are the same; in this case we return True
    if np.sign(np.dot(hyperplane1_unit_normal, point) + hyperplane1_bias) != np.sign(np.dot(hyperplane2_unit_normal, point) + hyperplane2_bias):
        return True
    
    # If the point is very close to one of the hyperplanes in the sense that evaluating a hyperplane at a point returns a number with absolute value less than or equal to the tolerance, we also say that the point is in between the hyperplanes; in this case we return True
    elif abs(np.dot(hyperplane1_unit_normal, point) + hyperplane1_bias) < tolerance or abs(np.dot(hyperplane2_unit_normal, point) + hyperplane2_bias) < tolerance:
        return True
    
    # If neither of the above conditions are satisfied, return False
    else:
        return False

''' Check if a given point is inside a given parallelepiped '''
def inside_parallelepiped(point, parallelepiped):
    # It is assumed that hyperplanes is a list of hyperplanes that make up the parallelepiped boundary, and that the list is given in order of parallel pairs hyperplanes
    hyperplanes = parallelepiped.hyperplane_list
    num_hyperplane_pairs = int(len(hyperplanes)/2)
    Bool_list = []

    # For every pair of hyperplanes, we check if the point is inbetween them
    for k in range(0, num_hyperplane_pairs):
        h1 = hyperplanes[2*k]
        h2 = hyperplanes[2*k+1]
        Bool_list.append(between_parallel_hyperplanes(point, h1, h2))

    return all(Bool_list)

''' Given an point and normal vector, return the hyperplane with that normal vector through the point '''
def make_hyperplane(point, normal_vector):
    # Let n denote the normal vector and b denote the bias
    # Then the hyperplane equation is given by <n, x> + b = 0, so b = - <n, x>
    return Hyperplane(normal_vector, -np.dot(normal_vector, point))

def binary_search_to_locate_point_within_hyperplanes(hyperplane_list, pt_hyperplane):
    pt_hyperplane_bias = pt_hyperplane.bias
    low = 0
    high = len(hyperplane_list) - 1
    mid = (low + high) // 2
    while low < high and low != mid and high != mid:
        if hyperplane_list[mid].bias <= pt_hyperplane_bias:
            low = mid
        elif hyperplane_list[mid].bias >= pt_hyperplane_bias:
            high = mid
        mid = (low + high) // 2
    return low

def sort_hyperplanes_by_bias(hyperplane_list):
    hyperplane_list.sort(key = lambda x: x.bias)
    return hyperplane_list

def find_cube_containing_point(sorted_hyperplane_dict, point, d):
    cube_index_list = []

    cube_hyperplane_list = []
    cube_hyperplane_normal_vectors_as_array = np.zeros((d*d, d))
    cube_bias_list = []

    for k in range(0, d):
        sorted_hyperplane_list = sorted_hyperplane_dict[k]
        hyperplane_through_point = make_hyperplane(point, sorted_hyperplane_list[0].normal_vec)
        index = binary_search_to_locate_point_within_hyperplanes(sorted_hyperplane_list, hyperplane_through_point)
        cube_index_list.append(index)
        h1 = sorted_hyperplane_list[cube_index_list[k]]
        h2 = sorted_hyperplane_list[cube_index_list[k]+1]

        cube_hyperplane_list.append(h1)
        cube_hyperplane_list.append(h2)
        cube_hyperplane_normal_vectors_as_array[k * 2] = h1.normal_vec
        cube_hyperplane_normal_vectors_as_array[(k * 2) + 1] = -h2.normal_vec
        cube_bias_list.append(-h1.bias)
        cube_bias_list.append(h2.bias)

    cube_bias_array = np.array(cube_bias_list)
    cube_as_polytope = pc.Polytope(cube_hyperplane_normal_vectors_as_array, cube_bias_array)

    if not (point in cube_as_polytope):
        print('ERROR')
    
    return tuple(cube_index_list)

def init_label_to_cube_dict(config):
    label_to_cubes_dict = dict()
    for label in range(config.num_labels + 1):
        label_to_cubes_dict[label] = []
    return label_to_cubes_dict

def get_domain_bounding_box(config):
    data_bounds_list = config.data_bounds
    return [data_bounds_list[i:i + 2] for i in range(0, len(data_bounds_list), 2)]

def get_domain_polytope(config):
    data_bounding_box = get_domain_bounding_box(config)
    return pc.box2poly(data_bounding_box)

def get_group_index_to_num_hyperplanes_dict(list_of_hyperplane_lists):
    num_of_hyperplanes_dict = dict()
    for j in range(0, len(list_of_hyperplane_lists)):
        num_of_hyperplanes_dict[j] = len(list_of_hyperplane_lists[j])
    return num_of_hyperplanes_dict

def get_cube_as_polytope(config, cube_id, sorted_hyperplane_dict):
    cube = tuple(cube_id.tolist())
    cube_hyperplane_list = []
    d = config.dimension
    cube_hyperplane_normal_vectors_as_array = np.zeros((d*d, d))
    cube_bias_list = []
    for i in range(0, d):
        h1 = sorted_hyperplane_dict[i][cube[i]]
        h2 = sorted_hyperplane_dict[i][cube[i]+1]
        cube_hyperplane_list.append(h1)
        cube_hyperplane_list.append(h2)
        cube_hyperplane_normal_vectors_as_array[i * 2] = h1.normal_vec
        cube_hyperplane_normal_vectors_as_array[(i * 2) + 1] = -h2.normal_vec
        cube_bias_list.append(-h1.bias)
        cube_bias_list.append(h2.bias)
    cube_bias_array = np.array(cube_bias_list)
    return pc.Polytope(cube_hyperplane_normal_vectors_as_array, cube_bias_array)

def get_cube_vertices_for_labeling(config, cube_as_polytope):
    domain = get_domain_polytope(config)
    # get the corners of the cube
    cube_as_polytope_intersected_with_domain = pc.intersect(cube_as_polytope, domain)

    vertices_to_consider_when_labeling_as_array = pc.extreme(cube_as_polytope_intersected_with_domain)
    if vertices_to_consider_when_labeling_as_array is not None:
        return vertices_to_consider_when_labeling_as_array.tolist()
    else:
        return []
    
def get_cube_label(config, vertex_list, model, labeling_threshold):
    label_list = []

    for vertex in vertex_list:
        # Check if vertex is already a tensor
        if not isinstance(vertex, torch.Tensor):
            # Convert vertex to a tensor
            vertex = torch.tensor(vertex)
        network_value_at_vertex = evaluate_regression_network(config, vertex, model)
        for label in range(config.num_labels):
            if label - labeling_threshold <= network_value_at_vertex <= label + labeling_threshold:
                label_list.append(label)
                break
            else:
                label_list.append(config.num_labels)

    label_set = list(set(label_list))
    if len(label_set) > 1:
        return config.num_labels
    elif len(label_set) == 1:
        return label_set[0]

def update_label_to_cubes_dict(label_to_cubes_dict, label, cube_id):
    labeled_cube_list = label_to_cubes_dict[label]
    labeled_cube_list.append(cube_id.tolist())
    label_to_cubes_dict[label] = labeled_cube_list
    return label_to_cubes_dict

def get_num_cubes_labeled(config, label_to_cubes_dict):
    num_cubes_labeled = 0
    for label in range(config.num_labels + 1):
        cube_list = label_to_cubes_dict[label]
        num_cubes_labeled += len(cube_list)
    return num_cubes_labeled

def get_labeled_cubes(config, sorted_hyperplane_dict, list_of_hyperplane_lists, model, labeling_threshold):

    # Define the polytope (cube) that represents the domain
    label_to_cubes_dict = init_label_to_cube_dict(config)

    num_of_hyperplanes_dict = get_group_index_to_num_hyperplanes_dict(list_of_hyperplane_lists)

    cube_id = torch.zeros(config.dimension, dtype = int)
    number_cubes = 0
    cubes_left_to_label = True
    is_max_number = [False]
    while cubes_left_to_label:
        cube_as_polytope = get_cube_as_polytope(config, cube_id, sorted_hyperplane_dict)
        vertex_list = get_cube_vertices_for_labeling(config, cube_as_polytope)
        if len(vertex_list) > 0:
            label = get_cube_label(config, vertex_list, model, labeling_threshold)
            label_to_cubes_dict = update_label_to_cubes_dict(label_to_cubes_dict, label, cube_id)
        
        number_cubes += 1

        if all(is_max_number):
            break

        new_cube_id = next_cube_index(cube_id, config.dimension, num_of_hyperplanes_dict)

        is_max_number = []
        for k in range(0, config.dimension):
            is_max_number.append(int(new_cube_id[k]) == num_of_hyperplanes_dict[k] - 2)

        if number_cubes % 1000 == 0 and config.verbose:
            print("Cubes labeled so far: ", number_cubes, flush = True)

        cube_id = new_cube_id

    return label_to_cubes_dict

def make_data_inside_cube_dict(data_as_tensors, sorted_hyperplane_dict, d):
    data_inside_cube_dict = dict()

    # For each datapoint, identify which cube the datapoint is inside


    for index, datapt in enumerate(data_as_tensors):
        index_of_cube_datapt_inside = tuple(find_cube_containing_point(sorted_hyperplane_dict, datapt, d))

        # Check if there are any datapoints already associated to that cube
        try:
            data_inside_cube_index_list = data_inside_cube_dict[index_of_cube_datapt_inside]

        # If not, initiate the list of data inside that cube as empty
        except:
            data_inside_cube_index_list = []

        # Add the index of the data point to the list
        data_inside_cube_index_list.append(index)

        # Update the dictionary
        data_inside_cube_dict[index_of_cube_datapt_inside] = data_inside_cube_index_list
    return data_inside_cube_dict

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
                if data_is_on_one_side_of_hyperplane(H, data_as_tensors, d):
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
        if config.verbose:
            print('Post-processing dimension: ', k)
        list_of_hyperplanes_with_boundaries, is_boundary_hyperplane_dict = add_boundaries(config, hyperplane_dict[k], data_as_tensors, is_boundary_hyperplane_dict)
        sorted_hyperplane_list_with_single_boundaries = sort_hyperplanes_and_remove_unecessary_hyperplanes(list_of_hyperplanes_with_boundaries, is_boundary_hyperplane_dict)
        list_of_hyperplane_lists.append(sorted_hyperplane_list_with_single_boundaries)
        total_hyperplane_list.extend(sorted_hyperplane_list_with_single_boundaries)
        sorted_hyperplane_dict[k] = sorted_hyperplane_list_with_single_boundaries
    
    return list_of_hyperplane_lists, sorted_hyperplane_dict, total_hyperplane_list

def get_homology_dict_from_model(config, N, model, data, job_index, labeling_threshold, batch_size = 1):
    cube_reg_model = load_model(N, config, job_index, batch_size)
    parameter_dict = get_model_parameters(cube_reg_model)

    coordinate_to_weights_dict = make_coordinate_to_weights_dict(config, parameter_dict["shared_weight_matrix"], N)

    data_as_tensors = convert_data_to_tensors(data, config.dimension)
    # To do: Compute whether a hyperplane is a boundary hyperplane using the domain polytope rather than the data
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
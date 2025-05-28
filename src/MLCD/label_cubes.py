import torch
from .decomposition import get_group_index_to_num_hyperplanes_dict, next_cube_index
from .cube import get_cube_as_polytope, get_cube_vertices_for_labeling, Cube

# Evaluate the regression network model at a single point x
def evaluate_regression_network(config, x, model):
    with torch.no_grad():
        d = config.dimension
        result = model(x.clone().detach().view(1, d))
    return float(result)

# Label a cube given a list of vertices, a regression network model, and a labeling threhsold
def get_cube_label(config, vertex_list, model, labeling_threshold):
    # Instantiate a list to keep track of the label of each vertex of the cube
    label_list = []

    for vertex in vertex_list:
        # Check if vertex is already a tensor
        if not isinstance(vertex, torch.Tensor):
            # Convert vertex to a tensor
            vertex = torch.tensor(vertex)
        # Evaluate the network at the vertex
        network_value_at_vertex = evaluate_regression_network(config, vertex, model)

        has_label = False
        
        # For each possible label, check if the network value is within epsilon of the label
        for label in range(config.num_labels):
            if label - labeling_threshold <= network_value_at_vertex <= label + labeling_threshold:
                # If the criterion is met, then break
                # Add the label to the label_list
                label_list.append(label)
                has_label = True
                break

        # If the network value is not within labeling_threshold of any of the labels, then give the uncertain label
        if not has_label:
            label_list.append(config.num_labels)

    # Check if the label vertices agree
    label_set = list(set(label_list))
    # If the label vertices do not agree, return the uncertain label
    if len(label_set) > 1:
        return config.num_labels
    # Otherwise, return the label at which they agree
    elif len(label_set) == 1:
        return label_set[0]

def update_label_to_cubes_dict(label_to_cubes_dict, label, cube_id):
    labeled_cube_list = label_to_cubes_dict[label]
    labeled_cube_list.append(cube_id.tolist())
    new_dict = dict()
    new_dict[label] = labeled_cube_list
    label_to_cubes_dict.update(new_dict)
    return label_to_cubes_dict

def get_num_cubes_labeled(config, label_to_cubes_dict):
    num_cubes_labeled = 0
    for label in range(config.num_labels + 1):
        cube_list = label_to_cubes_dict[label]
        num_cubes_labeled += len(cube_list)
    return num_cubes_labeled

def init_label_to_cube_dict(config):
    label_to_cubes_dict = dict()
    for label in range(config.num_labels + 1):
        label_to_cubes_dict[label] = []
    return label_to_cubes_dict

def get_labeled_cubes(config, sorted_hyperplane_dict, list_of_hyperplane_lists, model, labeling_threshold):

    # Define the polytope (cube) that represents the domain
    label_to_cubes_dict = init_label_to_cube_dict(config)

    num_of_hyperplanes_dict = get_group_index_to_num_hyperplanes_dict(list_of_hyperplane_lists)

    cube_list_for_polytope_figure = []

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
            if config.dimension == 2:
                new_cube = Cube(label, vertex_list)
                cube_list_for_polytope_figure.append(new_cube)
        
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

    return label_to_cubes_dict, cube_list_for_polytope_figure
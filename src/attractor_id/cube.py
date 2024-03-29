import numpy as np
from .domain import get_domain_polytope
import polytope as pc # It is also recommended to import cvxopt, which is a non-required dependency of polytope

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
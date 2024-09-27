import numpy as np
from .domain import get_domain_polytope
import polytope as pc # It is also recommended to import cvxopt, which is a non-required dependency of polytope

class Cube:
    def __init__(self, label, vertex_list):
        self.label = label
        self.vertex_list = vertex_list
        
def get_cube_as_polytope(config, cube_id, sorted_hyperplane_dict):
    cube = tuple(cube_id.tolist())
    cube_hyperplane_list = []
    d = config.dimension
    cube_hyperplane_normal_vectors_as_array = np.zeros((d*d, d))
    # To do: Why does this say d*d? 
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
    
    # Intersect the cube with the domain
    cube_as_polytope_intersected_with_domain = pc.intersect(cube_as_polytope, domain)

    # Compute the extreme points of the intersection
    # If the interseciton is empty, then pc.extreme returns None
    vertices_to_consider_when_labeling_as_array = pc.extreme(cube_as_polytope_intersected_with_domain)

    # If the intersection is non-empty, return a list of extreme points of the intersection
    if vertices_to_consider_when_labeling_as_array is not None:
        return vertices_to_consider_when_labeling_as_array.tolist()

    # Otherwise, return the empty list
    else:
        return []
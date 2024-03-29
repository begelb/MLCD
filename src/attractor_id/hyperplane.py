import numpy as np
import torch

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

''' Compute the intersection of d hyperplanes in general arrangement in dimension d '''
def hyperplane_intersection(hyperplane_list, d):
    N = torch.zeros(d, d)
    b = torch.zeros(d)
    for i in range(0, d):
        N[i] = torch.from_numpy(hyperplane_list[i].normal_vec)
        b[i] = -hyperplane_list[i].bias
    # solve and return the system of linear equations Nx = b
    return torch.linalg.solve(N, b)

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
    
''' Given a point and normal vector, return the hyperplane with that normal vector through the point '''
def make_hyperplane(point, normal_vector):
    # Let n denote the normal vector and b denote the bias
    # Then the hyperplane equation is given by <n, x> + b = 0, so b = - <n, x>
    return Hyperplane(normal_vector, -np.dot(normal_vector, point))
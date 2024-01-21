#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:27:17 2024

@author: paultatasciore
"""
import sys
sys.path.append("/usr/local/lib/python3.9/site-packages")
sys.path.append(
    "/Users/paultatasciore/Library/Python/3.9/lib/python/site-packages")
import systems
import iterate
import numpy as np
import gudhi
from format_data import save_formatted_data
import pickle


''' Global variables set by user '''

# system is an integer that refers to which dynamical system the user would like to use
system = 1


# number of points to be sampled and iterated in the domain
num_of_pts = 100


''' Global variables that should not be changed by the user '''
if system == 1:
    DS = systems.Straight
if system == 2:
    DS = systems.Curved
if system == 3:
    DS = systems.Radial2
if system == 4:
    DS = systems.Radial
if system == 5:
    DS = systems.Straight_4d
if system == 6:
    DS = systems.Curved_4d
if system == 7:
    DS = systems.EMT
if system == 8:
    DS = systems.Periodic_3d
    
domain = systems.systems[DS]
dim = len(domain)
step_size = 1 
eps = 1e-6
max_iter = 100

exp_info = {
    'system' : system,
    'num_of_pts' : num_of_pts,
    'domain' : domain,
    'dim' : dim,
    'step_size' : step_size,
    'eps' : eps,
    'max_iter' : max_iter
    }

if DS == systems.Radial or DS == systems.Radial2: # or DS == Ellipsoidal or DS == Ellipsoidal_3d:
    radial = True
else:
    radial = False

    
if __name__ == "__main__":
    X0 = iterate.init_pts(domain, num_of_pts) 
    X1, hausdorf_distances = iterate.remove_transience(DS, X0, step_size, max_iter, eps, domain, radial)  
    s0 = iterate.compute_norms(X1)
    M = 100 * len(hausdorf_distances)         
    X1, s = iterate.iter_and_compute_norms(DS, X1, M, step_size, domain, radial) 
    lifted_pts = iterate.make_lifted_pts(X1, s, domain, norm=False)
    

    diag = iterate.compute_persistance(lifted_pts) 
    gudhi.plot_persistence_diagram(diag)
    print(diag[:10])
    resolution = float(input("What is the resolution? ")) 
    labels = iterate.get_labels(lifted_pts, resolution)
    
    labeled_pts = iterate.make_labeled_pts(X0, X1, domain, labels)

    save_formatted_data(labeled_pts, num_of_pts)    
    #np.savetxt("data.csv", labeled_pts, delimiter=',')     # labeled_pts (initial points with labels)
    np.savetxt("images.csv", lifted_pts, delimiter=',')    # lifted_pts (final points with norms)
    
    with open('exp_info.pickle', 'wb') as handle:
        pickle.dump(exp_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

  
            

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:12:55 2024

@author: Paul Tatasciore
"""
import sys
sys.path.append("/usr/local/lib/python3.9/site-packages")
sys.path.append(
    "/Users/paultatasciore/Library/Python/3.9/lib/python/site-packages")
import numpy as np
from scipy.stats import qmc
from scipy.integrate import odeint
from scipy.spatial.distance import directed_hausdorff
import gudhi
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components



def make_grid(dim, num_points_per_dim, domain):
    x = [np.linspace(domain[i][0],domain[i][1], num_points_per_dim) for i in range(dim)]
    grid = np.array(np.meshgrid(*x)).T.reshape(-1,dim)   
    return grid

def init_pts(domain, num_of_pts, grid=False):
    dim = len(domain)
    if grid:
        X0 = make_grid(dim, 2**6+1, domain) 
    else:
        # Randomly sample the domain
        sample = qmc.LatinHypercube(d=dim).random(n=num_of_pts)
        X0 = qmc.scale(sample, [domain[i][0] for i in range(dim)], 
                        [domain[i][1] for i in range(dim)])
        
    return X0

def iterate_pts(DS, X0, t, domain, radial, stop_out_domain=False):  
    dim = len(domain)
    if radial:
        X0 = np.array([[np.sqrt(X0[i,0]**2 + X0[i,1]**2), np.arctan2(X0[i,1],X0[i,0])] for i in range(len(X0))])
    
    index_pts_in_domain = np.where(np.linalg.norm(X0, axis=1)<max(np.linalg.norm(domain, axis=1))*1.5)[0]
    
    if stop_out_domain:
        X1 = np.array([odeint(DS, X0[i], t) if i in index_pts_in_domain else odeint(DS, X0[i], t*0) for i in range(len(X0))]).reshape((len(X0),len(t),dim)).swapaxes(0,1) 

    else:
        X1 = np.array([odeint(DS, X0[i], t) for i in range(len(X0))]).reshape((len(X0),len(t),dim)).swapaxes(0,1) 
    
    if radial:
        X1 = np.array([[[X1[j,i,0] * np.cos(X1[j,i,1]), X1[j,i,0] * np.sin(X1[j,i,1])] for i in range(len(X0))] for j in range(len(X1))])
      
    return X1


def compute_norms(X1):
    norm = [np.sum(np.linalg.norm(X1[:,i])**2) for i in range(len(X1[0]))] 
    #norm = [np.array([np.linalg.norm(X1[i+1,j] - X1[i,j]) for i in range(len(X1)-1)]).sum() for j in range(len(X1[0]))]    
    s = np.expand_dims(np.array(norm), axis=1) 
    return s 

def iter_and_compute_norms(DS, X1, M, step_size, domain, radial=False):        
    t = np.linspace(0,M*step_size,M+1)    
    X1 = iterate_pts(DS, X1[-1], t, domain, radial)          
    s = compute_norms(X1)
    return X1, s

def hausdorf(X,Y):
    h1 = directed_hausdorff(X, Y)
    h2 = directed_hausdorff(Y, X)
    hf = max(h1, h2)
    return hf[0]

def remove_transience(DS, X, step_size, max_iter, eps, domain, radial):
    t = np.linspace(0,step_size,2)  
       
    X1 = iterate_pts(DS, X, t, domain, radial) 
    hausdorf_distances = [hausdorf(X1[0],X1[1])]
    X01 = [X1]
    while len(hausdorf_distances)<max_iter and hausdorf_distances[-1]>eps:   
        X1 = iterate_pts(DS, X1[1], t, domain, radial)
        hausdorf_distances.append(hausdorf(X1[0],X1[1]))
        X01.append(X1)
    return X1, hausdorf_distances

def compute_persistance(lifted_pts_in_domain):
    # Compute persistance homology and determine num of clusters / labels
    rips_complex = gudhi.RipsComplex(points=lifted_pts_in_domain, sparse=1)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=0)
    diag = simplex_tree.persistence()
    return diag

def get_labels(lifted_pts, resolution):
    tree = cKDTree(lifted_pts)
    edges = tree.sparse_distance_matrix(tree, max_distance=resolution, output_type="coo_matrix")
    n_components, labels = connected_components(csgraph=edges, directed=False, return_labels=True)
    print('There are ', n_components, 'connected components in the graph.')          
    return labels



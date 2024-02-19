#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:12:55 2024

@author: paultatasciore
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
import matplotlib.pyplot as plt




def make_grid(dim, num_points_per_dim, domain):
    x = [np.linspace(domain[i][0],domain[i][1], num_points_per_dim) for i in range(dim)]
    grid = np.array(np.meshgrid(*x)).T.reshape(-1,dim)   
    return grid

def init_pts(domain, num_of_pts, grid):
    dim = len(domain)
    if grid[0]:
        X0 = make_grid(dim, grid[1], domain) 
    else:
        # Randomly sample the domain
        sample = qmc.LatinHypercube(d=dim).random(n=num_of_pts)
        X0 = qmc.scale(sample, [domain[i][0] for i in range(dim)], 
                        [domain[i][1] for i in range(dim)])
        
    return X0

def iterate_DS_pts(DS, X0, t, domain, radial, stop_out_domain=False):  
    dim = len(domain)
    if radial:
        X0 = np.array([[np.sqrt(X0[i,0]**2 + X0[i,1]**2), np.arctan2(X0[i,1],X0[i,0])] for i in range(len(X0))])
    
    #index_in_domain = np.array([[domain[i][0] <= X0[j,:dim][i] <= domain[i][1] for i in range(dim)] for j in range(len(X0))]).all(1)
    index_in_domain = np.where(np.linalg.norm(X0, axis=1)<max(np.linalg.norm(domain, axis=1))*1.5)[0]
    
    if stop_out_domain:
        X1 = np.array([odeint(DS, X0[i], t) if i in index_in_domain else odeint(DS, X0[i], t*0) for i in range(len(X0))]).reshape((len(X0),len(t),dim)).swapaxes(0,1) 

    else:
        X1 = np.array([odeint(DS, X0[i], t) for i in range(len(X0))]).reshape((len(X0),len(t),dim)).swapaxes(0,1) 
    
    if radial:
        X1 = np.array([[[X1[j,i,0] * np.cos(X1[j,i,1]), X1[j,i,0] * np.sin(X1[j,i,1])] for i in range(len(X0))] for j in range(len(X1))])
      
    return X1

def iterate_MP_pts(MP, X0, domain):
    dim = len(domain)
    X_total = X0
    X = MP(X0, dim)
    X_total = np.vstack((X_total, X)).reshape(2, len(X0), dim)
    return X_total
    


def compute_norms(X1):
    norm = [np.sum(np.linalg.norm(X1[:,i])**2) for i in range(len(X1[0]))] 
    #norm = [np.array([np.linalg.norm(X1[i+1,j] - X1[i,j]) for i in range(len(X1)-1)]).sum() for j in range(len(X1[0]))]    
    s = np.expand_dims(np.array(norm), axis=1) 
    return s 

def hausdorf(X,Y):
    h1 = directed_hausdorff(X, Y)
    h2 = directed_hausdorff(Y, X)
    hf = max(h1, h2)
    return hf[0]

def compute_persistance(lifted_pts_in_domain):
    # Compute persistance homology and determine num of clusters / labels
    rips_complex = gudhi.RipsComplex(points=lifted_pts_in_domain, sparse=1)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=0)
    diag = simplex_tree.persistence()
    return diag

def get_labels(lifted_pts, resolution):
    #Labels = np.empty(len(lifted_pts))
    tree = cKDTree(lifted_pts)
    edges = tree.sparse_distance_matrix(tree, max_distance=resolution, output_type="coo_matrix")
    n_components, labels = connected_components(csgraph=edges, directed=False, return_labels=True)
    #Labels[np.where(lifted_pts[:,-1] != -1)] = labels
    #Labels[np.where(lifted_pts[:,-1] == -1)] = -1
    print('There are ', n_components, 'connected components in the graph.')          
    return labels, n_components

def make_lifted_pts(X1, s, norm, delay):
    # index_pts_in_domain = np.where(np.linalg.norm(X1[-1], axis=1)<=max(np.linalg.norm(domain, axis=1))*1.5)
    # index_pts_out_domain = np.where(np.linalg.norm(X1[-1], axis=1)>max(np.linalg.norm(domain, axis=1))*1.5)
    
    if delay[0]==True:
        lifted_pts = np.hstack([X1[-(i+1),:,:] for i in range(delay[1]+1)])
        
    if delay[0]==False:
        lifted_pts = X1[-1]

    if norm==True:
        lifted_pts = np.hstack((lifted_pts,s))
       
    return lifted_pts


def make_resolution(lifted_pts, dim, system, num_of_pts, path, useSugres):
    if lifted_pts.shape[1]==dim:
        diag = compute_persistance(lifted_pts) 
        fig = gudhi.plot_persistence_diagram(diag).figure
        fig.savefig(f'../output/figures/system{system}/' + "PD.jpg", bbox_inches='tight')
        plt.show()
        pd = [diag[i][1][1] for i in range(len(diag))]
        
        print("First 10 Elements of PD Diagram:")
        print("")
        print(pd[:10])
        
        data = pd[1:]
        median = np.median(data)
        std = np.std(data)
        
        threshold = useSugres[1]
        outliers = []
        zscore=[]
        for x in data:
            z_score = (x - median) / std
            zscore.append(z_score)
            if abs(z_score) > threshold:
                outliers.append(x)
        # print("")            
        # print("First 10 Elements of Zscore of PD:")
        # print("")            
        # print(zscore[:10])  
        try:
            sugres = outliers[-1] - .01
        except IndexError:
            print("")
            print("No Suggested Resolution. Try lowering sigma threshold.")
            sugres = 0
        print("")
        print("Suggested Resolution:", sugres)
        
        if useSugres[0]==False:
            resolution = float(input("Choose a resolution: ")) 
        else:
            resolution = sugres 
    else:
        diag = compute_persistance(lifted_pts) 
        d = [(0,(0.0,diag[i][1][1]/diag[1][1][1])) for i in range(len(diag))]
        fig = gudhi.plot_persistence_diagram(d).figure
        fig.savefig(f'../output/figures/system{system}/' + "PD.jpg", bbox_inches='tight')
        plt.show()      
        pd = [d[i][1][1] for i in range(len(d))]
        
        print("First 10 Elements of PD Diagram:")
        print("")
        print(pd[:10])
        
        data = pd[1:]
        median = np.median(data)
        std = np.std(data)
        
        threshold = useSugres[1]
        outliers = []
        zscore=[]
        for x in data:
            z_score = (x - median) / std
            zscore.append(z_score)
            if abs(z_score) > threshold:
                outliers.append(x)
        # print("")            
        # print("First 10 Elements of Zscore of PD:")
        # print("")            
        # print(zscore[:10])        
        try:
            sugres = outliers[-1] - .01
        except IndexError:
            print("")
            print("No Suggested Resolution. Try lowering sigma threshold.")
            sugres = 0

        print("")
        print("Suggested Resolution:", sugres)
        
        if useSugres[0]==False:
            resolution = float(input("Choose a resolution: ")) * diag[1][1][1]
        else:
            resolution = sugres * diag[1][1][1]
        
    return resolution, resolution/diag[1][1][1]


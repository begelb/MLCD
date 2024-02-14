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
from format_data import save_formatted_data
import pickle
import sys 
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import plot_data




''' Global variables set by user '''

systems_list = [1] 
num_of_pts = 100

for system in systems_list:
    
    norm = False
    delay = (False, 20)               # Delay coord with how many iterations back to include
    useSugres = (False, 3)            # Use Suggested Resolution based on how many std away from median (i.e sigma threshold)
    
    iter_grid = (False, 2**5+1)       # Sample points on grid with num_pts_per_dim or randomly 
    max_iter = 100
    maps = False
    save_full_orbit = False           # Save labeled_pts from entire orbit segment from transience 

    
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
    if system == 9:
        DS = systems.Ellipsoidal
    if system == 13:
        MP = systems.Leslie
        maps = True
        DS = []
        
    if maps:
        domain = systems.systems[MP]
    else:
        domain = systems.systems[DS]


    dim = len(domain)
    step_size = 1 
    eps = 1e-3
    
    exp_info = {
        'system' : system,
        'num_of_pts' : num_of_pts,
        'domain' : domain,
        'dim' : dim,
        'step_size' : step_size,
        'eps' : eps,
        'max_iter' : max_iter,
        'norm' : norm,
        'delay' : delay[0],
        'delay_iter' : delay[1]
        }
    
    if DS == systems.Radial or DS == systems.Radial2:# or DS == systems.Ellipsoidal:
        radial = True
    else:
        radial = False
    
    
    # if dne make one
    sys_path = '../data2/'    
    isExist = os.path.exists(sys_path)
    if not isExist:
       os.makedirs(sys_path)
       
    path = f'../data2/system{system}/'    
    isExist = os.path.exists(path)
    if not isExist:
       os.makedirs(path)

    pathfig = f'../output/figures/system{system}/'    
    isExist = os.path.exists(pathfig)
    if not isExist:
       os.makedirs(pathfig)
        
    X0 = iterate.init_pts(domain, num_of_pts, grid=iter_grid) 
    
    if DS == systems.Ellipsoidal: 
        
        a = 2 
        b = 1

        r_inv = R.from_euler('z', -45, degrees=True)
        Rotation_inv = r_inv.as_matrix()[:2, :2]
        I = np.identity(dim)
        I[0:2, 0:2] = Rotation_inv
        D_inv = np.array([[1/a,0], [0,1/b]])
        T_inv = D_inv @ Rotation_inv
        X0 = (T_inv @ X0.T).T
        
    if maps:
        X1 = iterate.iterate_MP_pts(MP, X0, domain)
        
    else:
        t = np.linspace(0,step_size,2)         
        X1 = iterate.iterate_DS_pts(DS, X0, t, domain, radial) 
    
   
    hausdorf_distances = [iterate.hausdorf(X1[0],X1[1])]
    X01 = [X1]
    while len(hausdorf_distances)<max_iter and hausdorf_distances[-1]>eps:  
        if maps:
            X1 = iterate.iterate_MP_pts(MP, X1[1], domain)
        else:
            X1 = iterate.iterate_DS_pts(DS, X1[1], t, domain, radial)
        hausdorf_distances.append(iterate.hausdorf(X1[0],X1[1]))
        X01.append(X1[1])
    
    
    if maps:
        X02 = []
        X2 = X1
        for i in range(100):
            X2 = iterate.iterate_MP_pts(MP, X2[-1], domain)
            X02.append(X2[1])
        X2= np.array(X02).reshape(100,num_of_pts,dim)
    else:
        M = 100 * len(hausdorf_distances)             
        t = np.linspace(0,M*step_size,M+1)    
        X2 = iterate.iterate_DS_pts(DS, X1[-1], t, domain, radial)          
        
    if DS == systems.Ellipsoidal: 
        r = R.from_euler('z', 45, degrees=True)
        Rotation = r.as_matrix()[:2, :2]
        I = np.identity(dim)
        I[0:2, 0:2] = Rotation
        D = np.array([[a,0], [0,b]])
        T = Rotation @ D  
        X0 = (T @ X0.T).T
        X1 = np.array([(T @ X1[i].T).T for i in range(len(X1))])
        X2 = np.array([(T @ X2[i].T).T for i in range(len(X2))])
    
    s = iterate.compute_norms(X2)
  
    lifted_pts = iterate.make_lifted_pts(X2, s, norm=norm, delay=delay) 
    index_in_domain = np.array([[domain[i][0] <= lifted_pts[j,:dim][i] <= domain[i][1] for i in range(dim)] for j in range(len(lifted_pts))]).all(1)
    lifted_pts[[not(i) for i in index_in_domain],-1] = -1 
    resolution, res = iterate.make_resolution(lifted_pts[index_in_domain], dim, system, num_of_pts, path, useSugres)
    labels, n_components = iterate.get_labels(lifted_pts[index_in_domain], resolution)
    
    if norm or delay[0]:
        resolution = res
    
    Labels = np.empty(len(lifted_pts))
    Labels[index_in_domain] = labels
    Labels[[not(i) for i in index_in_domain]] = -1
        
    ## Save labeled_pts from entire orbit segment from transience 
    if save_full_orbit:
        X11 = [X01[0][0,:,:], X01[0][1,:,:], *X01[1:]]
        X11 = np.array(X11).reshape(num_of_pts*(len(hausdorf_distances)+1),dim)
        labeled_pts = np.hstack((X11,np.tile(Labels, (len(hausdorf_distances)+1)).reshape(-1,1)))  

    ## Save labeled_pts from end points of orbit segment 
    else:
        labeled_pts = np.hstack((X0,Labels.reshape(-1,1)))

    exp_info['resolution'] = resolution
    exp_info['n_components'] = n_components
              
    save_formatted_data(labeled_pts, len(labeled_pts), path)    
    #np.savetxt(path + 'lifted.csv', lifted_pts, delimiter=',')   
    #np.savetxt(path + 'hausdorf_distances.csv', hausdorf_distances, delimiter=',')    
    
    with open(path + 'exp_info.pickle', 'wb') as handle:
        pickle.dump(exp_info, handle, protocol=pickle.HIGHEST_PROTOCOL)


    with open(f'../data2/system{system}/exp_info.pickle', 'rb') as handle:
        exp_info = pickle.load(handle)
        
        
    labeled_pts = np.loadtxt(f"../data2/system{system}/data.csv", delimiter=',')
    #lifted_pts = np.loadtxt(f"systems/{system}/{num_of_pts}pts/lifted.csv", delimiter=',')
    #hausdorf_distances = np.loadtxt(f"systems/{system}/{num_of_pts}pts/hausdorf_distances.csv", delimiter=',')


    dim = exp_info['dim']
    domain = exp_info['domain']
        
    fig1 = plt.figure()
    plt.plot(hausdorf_distances)
    plt.title('Hausdorff Disances Between Iterations')
    fig1.savefig(f"../output/figures/system{system}/Hausdorff_Disances_Between_Iterations.jpg", bbox_inches='tight')
    plt.show()

    index_in_domain = np.array([[domain[i][0] <= lifted_pts[j,:dim][i] <= domain[i][1] for i in range(dim)] for j in range(len(lifted_pts))]).all(1)
    
    
    plot_data.plot2d(labeled_pts[:num_of_pts][index_in_domain], labeled_pts[:num_of_pts][:,-1][index_in_domain], f"../output/figures/system{system}", system, title="Initial Points with Attractor Color")     
    plot_data.plot2d(lifted_pts[-num_of_pts:][index_in_domain], labeled_pts[-num_of_pts:][:,-1][index_in_domain], f"../output/figures/system{system}", system, title="Final Points with Attractor Color")

    #if dim==2 and lifted_pts.shape[1]!=dim:
        #plot3d(lifted_pts[-num_of_pts:][index_in_domain], labeled_pts[-num_of_pts:][:,-1][index_in_domain], f"systems/{system}/{num_of_pts}pts", title="Final Points with Attractor Color 3D") 
        #plot3d(labeled_pts[-num_of_pts:][index_in_domain], labeled_pts[-num_of_pts:][:,-1][index_in_domain], f"systems/{system}/{num_of_pts}pts", title="Initial Points with Attractor Color 3D")     
                

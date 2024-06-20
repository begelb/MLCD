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
import plot_data



''' Global variables set by user '''

systems_list = [1] 
for system in systems_list:
    num_of_pts = 1000
    
    norm = False
    delay = (False, 3)        # Delay coord with how many iterations back to include
    useSugres = (False, 3)    # Use Suggested Resolution based on how many std away from median (i.e sigma threshold)
    
    iter_grid = (False, 10)       # 2**5+1 Sample points on grid with num_pts_per_dim or randomly 
    max_iter = 100
    maps = False
    MP = None
    
    
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
    if system == 10:
        DS = systems.Periodic
    if system == 11:
        DS = systems.Periodic_back
    if system == 12:
        DS = systems.Torus
    if system == 13:
        MP = systems.Leslie
        maps = True
        DS = []
    if system == 14:
        MP = systems.Iris
        maps = True
        DS = []
    if system == 15:
        DS = systems.Lorentz
    if system == 16:
        DS = systems.FP3 
    if system == 17:
        DS = systems.Inf
    if system == 18:
        DS = systems.Memristive
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
    sys_path = 'systems/'    
    isExist = os.path.exists(sys_path)
    if not isExist:
       os.makedirs(sys_path)
       
    base_path = f'systems/{system}/'    
    isExist = os.path.exists(base_path)
    if not isExist:
       os.makedirs(base_path)
       
    path = f'systems/{system}/{num_of_pts}pts/'    
    isExist = os.path.exists(path)
    if not isExist:
       os.makedirs(path)
    
        
    X0 = iterate.init_pts(domain, num_of_pts, grid=iter_grid) 
    
    if maps:
        X1, acc = iterate.iterate_MP_pts(MP, X0, domain)
        #X1 = X1[:,:,0:7]
        
    else:
        t = np.linspace(0,step_size,2)         
        X1 = iterate.iterate_DS_pts(DS, X0, t, domain, radial) 
    
   
    hausdorf_distances = [iterate.hausdorf(X1[0],X1[1])]
    X01 = [X1]
    plot_data.plot2d_simple(X0, 'Iteration number 0')
    it = 1
    while len(hausdorf_distances)<max_iter and hausdorf_distances[-1]>eps:  
        if maps:
            X1, acc = iterate.iterate_MP_pts(MP, X1[1], domain)
        else:
            X1 = iterate.iterate_DS_pts(DS, X1[1], t, domain, radial)
            plot_data.plot2d_simple(X1[1], f'Iteration number {it}')
            it+=1
            
        hausdorf_distances.append(iterate.hausdorf(X1[0],X1[1]))
        X01.append(X1[1])
    
    
    if maps:
        X02 = []
        X2 = X1
        for i in range(100):
            X2, acc = iterate.iterate_MP_pts(MP, X2[-1], domain)
            X02.append(X2[1])#[:,0:7])
            
        X2= np.array(X02).reshape(100,num_of_pts,dim)
        

    else:
        M = 100 * len(hausdorf_distances)             
        t = np.linspace(0,M*step_size,M+1)    
        X2 = iterate.iterate_DS_pts(DS, X1[-1], t, domain, radial)          
    if MP == systems.Iris: 
        X3=X2    
        bias = X2[-1,:,7:]
        bias0 = X0[:,7:]
        X2 = X2[:,:,0:7]
        X1 = X1[:,:,0:7]
        X0 = X0[:,0:7]
        dim=7
    
    s = iterate.compute_norms(X2)
  
    lifted_pts = iterate.make_lifted_pts(X2, s, norm=norm, delay=delay) 
    index_in_domain = np.array([[domain[i][0]-100 <= lifted_pts[j,:dim][i] <= domain[i][1]+100 for i in range(dim)] for j in range(len(lifted_pts))]).all(1)
    lifted_pts[[not(i) for i in index_in_domain],-1] = -1 
    resolution, res = iterate.make_resolution(lifted_pts[index_in_domain], dim, system, num_of_pts, path, useSugres)
    labels, n_components = iterate.get_labels(lifted_pts[index_in_domain], resolution)
    
    if norm or delay[0]:
        resolution = res
    
    Labels = np.empty(len(lifted_pts))
    Labels[index_in_domain] = labels
    Labels[[not(i) for i in index_in_domain]] = -1
    
    ## Save labeled_pts from end points of orbit segment 
    labeled_pts = np.hstack((X0,Labels.reshape(-1,1)))
    
    ## Save labeled_pts from entire orbit segment from transience 
    # X11 = [X01[0][0,:,:], X01[0][1,:,:], *X01[1:]]
    # X11 = np.array(X11).reshape(num_of_pts*(len(hausdorf_distances)+1),dim)
    # labeled_pts = np.hstack((X11,np.tile(Labels, (len(hausdorf_distances)+1)).reshape(-1,1)))  

    exp_info['resolution'] = resolution
    exp_info['n_components'] = n_components
              
    save_formatted_data(labeled_pts, len(labeled_pts), path)    
    np.savetxt(path + 'lifted.csv', lifted_pts, delimiter=',')   
    np.savetxt(path + 'hausdorf_distances.csv', hausdorf_distances, delimiter=',')  

    
    if MP == systems.Iris:
        np.savetxt(path + 'acc.csv', np.array([acc,Labels]).T, delimiter=',')   

    
    with open(path + 'exp_info.pickle', 'wb') as handle:
        pickle.dump(exp_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
      
                

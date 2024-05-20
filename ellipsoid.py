#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:09:20 2023

@author: paultatasciore
"""


from scipy.stats import qmc
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R





dim = 2
num_of_pts = 10000
type = 'test'





def plot3d(pts, domain, labels, set_zlim=False, title="Points"):
    # Plot lifted dataset
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
    num_of_labels = int(max(labels) + 1)
    
    color_map = plt.get_cmap('Paired', num_of_labels)
    colors = [color_map(i) for i in range(num_of_labels)]

    for k, col in zip(range(num_of_labels), colors):
        my_members = labels == k
        ax.scatter3D(pts[my_members,0], pts[my_members,1], pts[my_members,2], color=col, alpha=0.1)
    #ax.set_xlim(domain[0][0], domain[0][1])
    #ax.set_ylim(domain[1][0], domain[1][1])
    if set_zlim:
        ax.set_zlim(domain[2][0], domain[2][1])  
    plt.title(title)

    plt.show()        
    return 


def plot2d(pts, domain, labels, title="Points", keep_scale=True):
    plt.figure()
    plt.clf()
     
    num_of_labels = int(max(labels) + 1)

    color_map = plt.get_cmap('Paired', num_of_labels)
    colors = [color_map(i) for i in range(num_of_labels)]
    
    for k, col in zip(range(num_of_labels), colors):
        my_members = labels == k
        plt.scatter(pts[my_members, 0], pts[my_members, 1], color=col)

    plt.title(title)
    if keep_scale:
        plt.xlim((domain[0][0], domain[0][1]))
        plt.ylim((domain[1][0], domain[1][1]))
    plt.show()


def init_pts(domain, num_of_pts, grid=False):
    dim = len(domain)
    sample = qmc.LatinHypercube(d=dim).random(n=num_of_pts)
    X1 = qmc.scale(sample, [domain[i][0] for i in range(dim)], 
                    [domain[i][1] for i in range(dim)])
    
    X2 = np.random.uniform(-2, 2, (int(num_of_pts/10),dim))
    X2 = X2[np.where(np.linalg.norm(X2, axis=1) < 2)]
    
    X0 = np.vstack((X1,X2))
    return X0


domain_ellipsoidal = ((-8,8),) * dim
X0 = init_pts(domain_ellipsoidal, num_of_pts)
pts_in_sphere = np.where(np.linalg.norm(X0, axis=1) < 2)[0] 
labeled_pts = np.vstack([np.hstack((X0[i],np.ones((1)))) if i in pts_in_sphere else np.hstack((X0[i],np.zeros((1)))) for i in range(len(X0))])
labeled_pts2 = np.copy(labeled_pts)
labeled_pts2[:,0] *= 2
r = R.from_euler('z', 45, degrees=True)
Rotation = r.as_matrix()[:2, :2]
I = np.identity(dim)
I[0:2, 0:2] = Rotation
Rotation = I
labeled_pts3 = np.hstack([(Rotation @ labeled_pts2[:,0:dim].T).T, labeled_pts2[:,-1].reshape(-1,1)])        
labeled_pts4 = np.array([pt for pt in labeled_pts3 if all(np.abs(pt)<4)])

l = labeled_pts4
while len(l) < num_of_pts:
    X0 = init_pts(domain_ellipsoidal, num_of_pts)

    pts_in_sphere = np.where(np.linalg.norm(X0, axis=1) < 2)[0] 

    labeled_pts = np.vstack([np.hstack((X0[i],np.ones((1)))) if i in pts_in_sphere else np.hstack((X0[i],np.zeros((1)))) for i in range(len(X0))])

    labeled_pts2 = np.copy(labeled_pts)
    labeled_pts2[:,0] *= 2

    r = R.from_euler('z', 45, degrees=True)
    Rotation = r.as_matrix()[:2, :2]

    I = np.identity(dim)
    I[0:2, 0:2] = Rotation
    Rotation = I

    labeled_pts3 = np.hstack([(Rotation @ labeled_pts2[:,0:dim].T).T, labeled_pts2[:,-1].reshape(-1,1)])        
    labeled_pts4 = np.array([pt for pt in labeled_pts3 if all(np.abs(pt)<4)])
    if len(labeled_pts4) != 0:
        l = np.vstack((l, labeled_pts4))

l = l[:num_of_pts]
np.random.shuffle(l)
#plot3d(l, domain_ellipsoidal, l[:,-1])   
#plot2d(l, domain_ellipsoidal, l[:,-1])    

np.savetxt(f"data/ellipsoidal_{dim}d/nf_{type}.csv", l) 

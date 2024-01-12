#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 22:17:24 2024

@author: paultatasciore
"""


import numpy as np 
import matplotlib.pyplot as plt


labeled_pts = np.loadtxt("data.csv")
lifted_pts = np.loadtxt("images.csv")




def plot2d(pts, labels, title="Points"):
    plt.figure()
    plt.clf()
     
    num_of_labels = int(max(labels) + 1)

    color_map = plt.get_cmap('Paired', num_of_labels)
    colors = [color_map(i) for i in range(num_of_labels)]
    
    for k, col in zip(range(num_of_labels), colors):
        my_members = labels == k
        plt.scatter(pts[my_members, 0], pts[my_members, 1], color=col)

    plt.title(title)
    plt.show()
          
def plot3d(pts, labels, title="Points"):
    # Plot lifted dataset
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    num_of_labels = int(max(labels) + 1)
    
    color_map = plt.get_cmap('Paired', num_of_labels)
    colors = [color_map(i) for i in range(num_of_labels)]

    for k, col in zip(range(num_of_labels), colors):
        my_members = labels == k

        ax.scatter3D(pts[my_members,0], pts[my_members,1], pts[my_members,2], color=col)

    plt.title(title)
    plt.show()        
    return 

plot2d(labeled_pts, labeled_pts[:,-1], title="Initial Points with Attractor Color")     
plot2d(lifted_pts, labeled_pts[:,-1], title="Final Points with Attractor Color")
#plot3d(lifted_pts_in_domain, domain, labels, title="Final Points with Attractor Color")   
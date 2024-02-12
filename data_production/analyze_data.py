#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 23:13:26 2024

@author: paultatasciore
"""

import numpy as np
import iterate
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
import sys 
import plot_data
import os 




def train_classifier(labeled_pts):
    #X_train, X_test, y_train, y_test = train_test_split(
     #   labeled_pts[:,0:-1], labeled_pts[:,-1], test_size=0.1)
    
    X_train, y_train = labeled_pts[:,0:-1], labeled_pts[:,-1]

    # MLP Classifier
    model = MLPClassifier(hidden_layer_sizes=(
        10, 10), activation='relu', learning_rate_init=0.001, max_iter=10000)
    model.fit(X_train, y_train)
    #score = model.score(X_test, y_test)
    #print("Score on test set:", score)
    #print(confusion_matrix(y_test, model.predict(X_test)))
    return model

if __name__ == "__main__":

    system = int(sys.argv[1])
    num_of_pts = int(sys.argv[2])
    num_points_per_dim = int(sys.argv[3]) #6 #int(np.round((6**6)**(1/dim)))  

    path = f'../data2/system{system}/'

    with open(path + 'exp_info.pickle', 'rb') as handle:
        exp_info = pickle.load(handle)
            
    labeled_pts = np.loadtxt(path + "data.csv", delimiter=',')
    domain = exp_info['domain']
    dim = exp_info['dim']
    
      
    grid_pts = iterate.make_grid(dim, num_points_per_dim, domain)
    

    model = train_classifier(labeled_pts)
    prob = model.predict_proba(grid_pts)
    labels_grid = np.argmax(prob, axis=1)
    labeled_pts_on_grid = np.hstack((grid_pts, labels_grid.reshape(-1,1)))
    
    path_out = path + str(num_points_per_dim) + 'gppd/'  
    isExist = os.path.exists(path_out)
    if not isExist:
       os.makedirs(path_out)
       
    if dim<3:
        #plot_data.plot2d(labeled_pts_on_grid, labeled_pts_on_grid[:,-1], path_out, system, title="Prediction on Grid from NN")
        
        plot_data.plot2dNNpred(model, grid_pts, labels_grid, f'../output/figures/system{system}/', system, title="Basins of Attraction and Seperatrix from Neural Network")        
        #plot_data.plotNNpred(model, grid_pts, path_out, title="Points")
    
    np.savetxt(path_out + "data_on_grid.csv", labeled_pts_on_grid, delimiter=',')     # labeled_pts (initial points with labels)


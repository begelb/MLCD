#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 22:17:24 2024

@author: paultatasciore
"""


import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import sys 
import pickle

def plotNNpred(model, grid_pts, path, title="Points"):
        c0 = np.min(model.predict_proba(grid_pts), axis=1)
        c0 = c0/np.max(c0)
        fig = plt.figure()
        plt.scatter(grid_pts[:, 0], grid_pts[:, 1], c=c0, cmap='Greys')
        plt.title("Approximate Basins of Attraction from Neural Network")
        plt.colorbar()
        t = title.replace(" ", "_")
        fig.savefig(str(path) + f"/{t}.jpg", bbox_inches='tight')
        plt.show()

def plot2dNNpred(model, pts, labels, path, system, title="Points"):
    fig = plt.figure()
    plt.clf()
     
    num_of_labels = int(max(labels) + 1)
    color_map = plt.get_cmap('Paired', num_of_labels)
    colors = [color_map(i) for i in range(num_of_labels)]
    
    for k, col in zip(range(num_of_labels), colors):
        my_members = labels == k
        plt.scatter(pts[my_members, 0], pts[my_members, 1], color=col)
        
    c0 = np.min(model.predict_proba(pts), axis=1)
    c0 = c0/np.max(c0)
    
    c_white = matplotlib.colors.colorConverter.to_rgba('white',alpha = 0)
    c_black= matplotlib.colors.colorConverter.to_rgba('white',alpha = 1)
    cmap_rb = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white,c_black],512)

    #plt.scatter(pts[:, 0], pts[:, 1], c=c0, cmap=cmap_rb)
    #plt.colorbar()

    plt.title(title)
    t = title.replace(" ", "_")
    fig.savefig(str(path) + f"/{t}.jpg", bbox_inches='tight')

    plt.show()

def plot2d(pts, labels, path, system, lifted, acc, title="Points"):

     
    num_of_labels = int(max(labels) + 1)

    color_map = plt.get_cmap('Paired', num_of_labels)
    colors = [color_map(i) for i in range(num_of_labels)]
    
    for i in range(1):
        fig = plt.figure()
        plt.clf()    
        for k, col in zip(range(num_of_labels), colors):
            my_members = labels == k
            plt.scatter(pts[my_members, i], pts[my_members, i+1], color=col)
    
        plt.title(title)
        t = title.replace(" ", "_")
        #plt.xlim(-2,2)
        #plt.ylim(-3.5,3.5)
        fig.savefig(str(path) + f"/{t}.jpg", bbox_inches='tight')

    color_map = plt.get_cmap('Paired', 3)
    colors = [color_map(i) for i in range(3)]
    # plt.scatter(lifted[0,0], lifted[0,1], color=colors[0], marker='*', edgecolor='black', s=100)
    # plt.scatter(lifted[1,0], lifted[1,1], color=colors[1], marker='*', edgecolor='black', s=100)
    # plt.scatter(45, 14, color='k', marker='*', edgecolor='black', s=100)
    # plt.scatter(21, 3, color='k', marker='*', edgecolor='black', s=100)
    # plt.scatter(3, 31, color='k', marker='*', edgecolor='black', s=100)
    # acc1 = np.round(np.average(acc[:,0][np.where(acc[:,1]==0)]),2)
    # acc2 = np.round(np.average(acc[:,0][np.where(acc[:,1]==1)]),2)
    
    # std1 = np.round(np.std(acc[:,0][np.where(acc[:,1]==0)]),2)
    # std2 = np.round(np.std(acc[:,0][np.where(acc[:,1]==1)]),2)

    #plt.legend([colors[0], colors[1]], labels=[f'acc = {acc1}, std = {std1}', f'acc = {acc2}, std = {std2}'], loc="lower left")
    fig.savefig(str(path) + f"/{t}.jpg", bbox_inches='tight')
    plt.show()
    
def plot2d_simple(pts, title="Points"):


    plt.scatter(pts[:, 0], pts[:, 1], color='k')

    plt.title(title)
    plt.xlim(-2,2)
    plt.ylim(-3.5,3.5)
    plt.show()  
        
def plot3d(pts, labels, path, title="Points"):
    # Plot lifted dataset
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    num_of_labels = int(max(labels) + 1)
    
    color_map = plt.get_cmap('Paired', num_of_labels)
    colors = [color_map(i) for i in range(num_of_labels)]

    for k, col in zip(range(num_of_labels), colors):
        my_members = labels == k

        ax.scatter3D(pts[my_members,0], pts[my_members,1], pts[my_members,2] / np.max(pts[:,-1]), color=col)

    plt.title(title)
    t = title.replace(" ", "_")
    fig.savefig(str(path) + f"/{t}.jpg", bbox_inches='tight')
    plt.show()        
    return 


if __name__ == "__main__":
    system = int(sys.argv[1])
    num_of_pts = int(sys.argv[2])


    with open(f'systems/{system}/{num_of_pts}pts/exp_info.pickle', 'rb') as handle:
        exp_info = pickle.load(handle)
        
        
    labeled_pts = np.loadtxt(f"systems/{system}/{num_of_pts}pts/data.csv", delimiter=',')
    lifted_pts = np.loadtxt(f"systems/{system}/{num_of_pts}pts/lifted.csv", delimiter=',')
    hausdorf_distances = np.loadtxt(f"systems/{system}/{num_of_pts}pts/hausdorf_distances.csv", delimiter=',')
    if system == 14:
        acc = np.loadtxt(f"systems/{system}/{num_of_pts}pts/acc.csv", delimiter=',')
    else:
        acc = None


    dim = exp_info['dim']
    domain = exp_info['domain']


    fig1 = plt.figure()
    plt.plot(hausdorf_distances)
    #plt.ylim([0, 3])
    plt.title('Hausdorff Disances Between Iterations')
    fig1.savefig(f"systems/{system}/{num_of_pts}pts/Hausdorff_Disances_Between_Iterations.jpg", bbox_inches='tight')
    plt.show()

    index_in_domain = np.array([[domain[i][0]-100 <= lifted_pts[j,:dim][i] <= domain[i][1]+100 for i in range(dim)] for j in range(len(lifted_pts))]).all(1)
    
    
    plot2d(labeled_pts[:num_of_pts][index_in_domain], labeled_pts[:num_of_pts][:,-1][index_in_domain], f"systems/{system}/{num_of_pts}pts", system, lifted_pts, acc, title="Initial Points with Attractor Color")     
    plot2d(lifted_pts[-num_of_pts:][index_in_domain], labeled_pts[-num_of_pts:][:,-1][index_in_domain], f"systems/{system}/{num_of_pts}pts", system, lifted_pts, acc, title="Final Points with Attractor Color")

    if dim==3 and lifted_pts.shape[1]!=dim:
        plot3d(lifted_pts[-num_of_pts:][index_in_domain], labeled_pts[-num_of_pts:][:,-1][index_in_domain], f"systems/{system}/{num_of_pts}pts", title="Final Points with Attractor Color 3D") 
        plot3d(labeled_pts[-num_of_pts:][index_in_domain], labeled_pts[-num_of_pts:][:,-1][index_in_domain], f"systems/{system}/{num_of_pts}pts", title="Initial Points with Attractor Color 3D")     

    # plt.violinplot(lifted_pts)
    # for i in range(len(lifted_pts[0])):
    #     plt.figure()
    #     plt.violinplot(lifted_pts[:,i])
        
    # j=10
    # pd_sum = np.empty(j)
    # init_number_of_pts = int(len(lifted_pts) / js)
    
    # for i in range(j):
    
    #     if lifted_pts.shape[1]==dim:
    #         diag = iterate.compute_persistance(lifted_pts[index_in_domain][0:init_number_of_pts*(i+1),:]) 
    #         pd_sum[i] = sum([diag[n][1][1] for n in range(1,len(diag))])/len(diag)
    
    #     else:
    #         diag = iterate.compute_persistance(lifted_pts[index_in_domain][0:init_number_of_pts*(i+1),:]) 
    #         d = [(0,(0.0,diag[i][1][1]/diag[1][1][1])) for i in range(len(diag))]
    #         pd_sum[i] = sum([d[n][1][1] for n in range(1,len(d))])/len(d)
    color_map = plt.get_cmap('Paired', 3)
    colors = [color_map(i) for i in range(3)]

    # fig, ax1 = plt.subplots()
    # violin1 = ax1.violinplot(lifted_pts[np.where(labeled_pts[:,-1]==0)])
    # violin2 = ax1.violinplot(lifted_pts[np.where(labeled_pts[:,-1]==1)])
    # for pc in violin1["bodies"]:
    #     pc.set_facecolor(colors[0])
    # for pc1 in violin2["bodies"]:
    #     pc1.set_facecolor(colors[2])
    # acc1 = np.round(np.average(acc[:,0][np.where(acc[:,1]==0)]),2)
    # acc2 = np.round(np.average(acc[:,0][np.where(acc[:,1]==1)]),2)

    # std1 = np.round(np.std(acc[:,0][np.where(acc[:,1]==0)]),2)
    # std2 = np.round(np.std(acc[:,0][np.where(acc[:,1]==1)]),2)

    # #plt.legend(labels=[f'acc = {acc1}, std = {std1}', f'acc = {acc2}, std = {std2}'], loc="lower left")
    # plt.legend(labelcolor=[colors[0], colors[2]], labels=[f'acc = {acc1}, std = {std1}', f'acc = {acc2}, std = {std2}'], loc="lower left")
    # plt.title('Violin Plot for each Weight of Neural Network')
    # fig.savefig(str(f"systems/{system}/{num_of_pts}pts") + "violin.jpg", bbox_inches='tight')
    # plt.show()
    
    # fig3 = plt.figure()            
    # plt.plot([init_number_of_pts * (i+1) for i in range(len(pd_sum))], pd_sum)
    # plt.xlabel("Number of Points")
    # plt.title("Ave Death Time of Finite PD Elements")
    # fig3.savefig(f"systems/{system}/{num_of_pts}pts/Ave_Death_Time_of_Finite_PD_Elements.jpg", bbox_inches='tight')
    # plt.show()


    # fig, ax1 = plt.subplots()
    # violin1 = ax1.violinplot(labeled_pts[np.where(labeled_pts[:,-1]==0)])
    # violin2 = ax1.violinplot(labeled_pts[np.where(labeled_pts[:,-1]==1)])
    # for pc in violin1["bodies"]:
    #     pc.set_facecolor(colors[0])
    # for pc1 in violin2["bodies"]:
    #     pc1.set_facecolor(colors[2])
    # acc1 = np.round(np.average(acc[:,0][np.where(acc[:,1]==0)]),2)
    # acc2 = np.round(np.average(acc[:,0][np.where(acc[:,1]==1)]),2)

    # std1 = np.round(np.std(acc[:,0][np.where(acc[:,1]==0)]),2)
    # std2 = np.round(np.std(acc[:,0][np.where(acc[:,1]==1)]),2)

    #plt.legend(labels=[f'acc = {acc1}, std = {std1}', f'acc = {acc2}, std = {std2}'], loc="lower left")
    #plt.legend([colors[0], colors[2]], labels=[f'acc = {acc1}, std = {std1}', f'acc = {acc2}, std = {std2}'], loc="lower left")
    #plt.title('Violin Plot for Weights of Neural Network')
    
    
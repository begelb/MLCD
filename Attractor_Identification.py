#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 22:29:39 2023

@author: paultatasciore
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 12:18:07 2023

@author: paultatasciore
"""

import sys
sys.path.append("/usr/local/lib/python3.9/site-packages")
sys.path.append(
    "/Users/paultatasciore/Library/Python/3.9/lib/python/site-packages")
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import qmc
from ripser import ripser
from persim import plot_diagrams
from scipy.spatial.distance import directed_hausdorff
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, OPTICS
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import confusion_matrix
from sklearn import decomposition
import gudhi
from scipy.integrate import solve_ivp
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
        # Random sample the domain
        sample = qmc.LatinHypercube(d=dim).random(n=num_of_pts)
        X0 = qmc.scale(sample, [domain[i][0] for i in range(dim)], 
                        [domain[i][1] for i in range(dim)])
        
    return X0

def iterate_pts(DS, X0, t, domain, radial):  
    dim = len(domain)
    if radial:
        X0 = np.array([[np.sqrt(X0[i,0]**2 + X0[i,1]**2), np.arctan2(X0[i,1],X0[i,0])] for i in range(len(X0))])
    
    index_pts_in_domain = np.where(np.linalg.norm(X0, axis=1)<max(np.linalg.norm(domain, axis=1))*1.5)[0]
    X1 = np.array([odeint(DS, X0[i], t) if i in index_pts_in_domain else odeint(DS, X0[i], t*0) for i in range(len(X0))]).reshape((len(X0),len(t),dim)).swapaxes(0,1) 

    #XX1 = np.array([solve_ivp(DS, t, X0[i], method='RK45', rtol=1e-10, atol=1e-10) for i in w]).reshape((len(w),len(t),dim)).swapaxes(0,1) 
     
    if radial:
        X1 = np.array([[[X1[j,i,0] * np.cos(X1[j,i,1]), X1[j,i,0] * np.sin(X1[j,i,1])] for i in range(len(X0))] for j in range(len(X1))])
      
    return X1

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
        
def apply_pca(lifted_pts):
    plt.cla()
    pca = decomposition.PCA(n_components=2)
    pca.fit(lifted_pts)
    lifted_pts = pca.transform(lifted_pts)
    print(pca.explained_variance_ratio_)
    return lifted_pts

def compute_persistance(lifted_pts_in_domain):
    # Compute persistance homology and determine num of clusters / labels
    rips_complex = gudhi.RipsComplex(points=lifted_pts_in_domain, sparse=1)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=0)
    diag = simplex_tree.persistence()
    return diag

def compute_kmeans(lifted_pts_in_domain, num_clusters):
    # Compute labeled ds via kmeans
    km = KMeans(num_clusters, init='k-means++', verbose=0) 
    km.fit(lifted_pts_in_domain)
    labels = km.predict(lifted_pts_in_domain)
    #cluster_centers = km.cluster_centers_
    return labels

def get_labels(lifted_pts, resolution):
    tree = cKDTree(lifted_pts)
    edges = tree.sparse_distance_matrix(tree, max_distance=resolution, output_type="coo_matrix")
    n_components, labels = connected_components(csgraph=edges, directed=False, return_labels=True)
    print('There are ', n_components, 'connected components in the graph.')          
    return labels

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
      
def train_classifier(labeled_pts):
    X_train, X_test, y_train, y_test = train_test_split(
        labeled_pts[:,0:-1], labeled_pts[:,-1], test_size=0.2)

    # MLP Classifier
    model = MLPClassifier(hidden_layer_sizes=(
        100, 100), activation='relu', learning_rate_init=0.001, max_iter=1000)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print("Score on test set:", score)
    print(confusion_matrix(y_test, model.predict(X_test)))
    return model

def train_regressor(labeled_pts):
    X_train, X_test, y_train, y_test = train_test_split(
        labeled_pts[:,0:-1], labeled_pts[:,-1], test_size=0.2)

    # MLP Classifier
    model = MLPRegressor(hidden_layer_sizes=(
        100,), max_iter=1000, verbose=True, solver='sgd')
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print("Score on test set:", score)
    return model
       

domain_duffing = ((-10,10),(-10,10))
def Duffing(y, t):
    omega0 = 1
    omega = 1.5
    epsilon = 1
    alpha = 0.03
    delta = 0.2
    gamma = 2.5
    y0 = y[1]
    y1 = -omega0**2 * y[0] + epsilon * ( -delta * y[1] -  alpha * y[0]**3 + gamma * np.cos(omega * t) ) 
    dydt = np.array([y0, y1])
    return dydt

domain_harmonic = ((-10,10),(-10,10))
def Harmonic(y, t):
    omega0 = 1
    y0 = y[1]
    y1 = -omega0**2 * y[0] - y[1]
    dydt = np.array([y0, y1])
    return dydt

domain_radial = ((-5,5),(-5,5))
def Radial(y, t):
    # Radial System
    theta = 1
    r = - y[0] * (y[0]-1) * (y[0]-2) * (y[0]-3) * (y[0]-4) 
    dydt = np.array([r,theta])
    return dydt

domain_radial2 = ((-4,4),(-4,4))
def Radial2(y, t):
    # Radial System
    theta = 1
    r = - y[0] * (y[0]-1) * (y[0]-2) * (y[0]-3) 
    dydt = np.array([r,theta])
    return dydt

domain_ellipsoidal = ((-8,8),(-8,8))
def Ellipsoidal(y, t):
    # Radial System
    theta = 1
    r = - y[0] * (y[0]-1) * (y[0]-2) * (y[0]-3) 
    dydt = np.array([r,theta])
    return dydt

domain_ellipsoidal3d = ((-8,8),(-8,8),(-8,8))
def Ellipsoidal_3d(y, t):
    theta = 1
    r = - y[0] * (y[0]-1) * (y[0]-2) * (y[0]-3) 
    phi = np.sqrt(2)/2
    dydt = np.array([r,theta,phi])
    return dydt

domain_straight = ((-2,2),(-3.5,3.5))
def Straight(y, t):
    # Kalies Simple Example
    theta=np.pi/3
    u=[np.cos(theta)*y[0]+np.sin(theta)*y[1], -np.sin(theta)*y[0]+np.cos(theta)*y[1]]
    ud=u[0]*(1-u[0]**2)
    vd=(u[0]**2)*(3-2*u[0]**2)-u[1]        
    dydt=[np.cos(theta)*ud-np.sin(theta)*vd, np.sin(theta)*ud+np.cos(theta)*vd]

    return dydt

#domain_curved = ((-2,2),(-3.5,3.5),(-2,2),(-2,2),(-2,2),(-2,2))   
#domain_curved = ((-2,2),(-3.5,3.5),(-2,2),(-2,2))   
domain_curved = ((-2,2),(-3.5,3.5))    
def Curved(y, t):   
    # Kalies more comp Example (4d)      
      p=1
      e=3   
      ud = -y[0]
      vd = (y[1] - p * y[0]**2) * (e**2 - y[1]**2)
      dydt = [ud, vd]

      #wd = -y[2]
      #xd = -y[3]
      #dydt = [ud, vd, wd, xd]
      
      #wd = -y[2]
      #xd = -y[3]
      #yd = -y[4]
      #zd = -y[5]
      
      #dydt = [ud, vd, wd, xd, yd, zd]

      return dydt

domain_periodic = ((0,7),(0.5,6))
#domain = ((3,4),(2,3))
def Periodic(y,t):
    # Marcio Example 1   
    r = 1
    K = 10
    m = 0.54
    a = 1.25
    b = -1.65
    s = 0.4
    y0 = r * y[0] * (1 - y[0] / K) - (m * y[1] * y[0]**2) /(a * y[0]**2 + b * y[0] + 1)
    y1 = s * y[1] * (1 - y[1] / y[0])
    dydt = np.array([y0, y1])
    return dydt


domain_straight4d = ((-2,2),(-3.5,3.5),(-2,2),(-2,2))    
def Straight_4d(y, t):    
    theta=np.pi/3
    u=[np.cos(theta)*y[0]+np.sin(theta)*y[1], -np.sin(theta)*y[0]+np.cos(theta)*y[1]]
    ud=u[0]*(1-u[0]**2)
    vd=(u[0]**2)*(3-2*u[0]**2)-u[1]        
    dydt=[np.cos(theta)*ud-np.sin(theta)*vd, np.sin(theta)*ud+np.cos(theta)*vd]
    
    wd = -y[2]
    xd = -y[3]
    dydt=[np.cos(theta)*ud-np.sin(theta)*vd, np.sin(theta)*ud+np.cos(theta)*vd, wd, xd]
    return dydt


#domain = ((0,149),(0,323),(0,459),(0,189),(0,397),(0,355))
domain_emt = ((0,13),(0,1.6),(0,1),(0,2.2),(0,6.7),(0,0.8))
e = 5
domain_emt = ((0.0222807-e,0.0222807+e),(3.04674-e,3.04674+e),(0.158407-e,0.158407+e),(3.71994-e,3.71994+e),(0.558874-e,0.558874+e),(3.6113-e,3.6113+e))
def EMT(X, t):
    # Define Hill model for network
    
    def H_minus(x, L, U, theta, n):
        return L + (U - L) * theta**n / (x**n + theta**n)
    
    def H_plus(x, L, U, theta, n):
        return L + (U - L) * x**n / (x**n + theta**n)


    # Number stable equilibria: 7
    # L = [[0.0, 0.0, 0.3474646378541038, 0.0, 0.0, 0.0], [0.22946501949510792, 0.0, 0.0, 0.0, 0.8298842391205155, 0.0], [0.0, 0.20730250967065328, 0.5730909762536707, 0.0, 0.6294329948114906, 0.09287730239753585], [0.7539385885881486, 0.0, 0.0, 0.0, 1.2165100030204878, 0.0], [0.0, 0.024937554151497357, 0.0, 0.2078369129941993, 0.0, 0.7775334176771063], [0.0, 0.0, 0.34034327948893484, 0.0, 0.0, 0.0]]
    # U = [[0.0, 0.0, 1.4125585768022326, 0.0, 0.0, 0.0], [1.4160852722012418, 0.0, 0.0, 0.0, 1.656637612206524, 0.0], [0.0, 1.5932066942490004, 2.0636567452371875, 0.0, 1.0591654192748299, 0.9592065405836583], [3.14564083069374, 0.0, 0.0, 0.0, 2.9783315814648392, 0.0], [0.0, 2.036233114864445, 0.0, 2.0020221352008707, 0.0, 1.4026790448670654], [0.0, 0.0, 0.9493533864654984, 0.0, 0.0, 0.0]]
    # T = [[0.0, 0.0, 0.3032286582407538, 0.0, 0.0, 0.0], [0.6308803819284967, 0.0, 0.0, 0.0, 0.17265397397047338, 0.0], [0.0, 0.37679586396207027, 1.8658142242897355, 0.0, 0.16941487491197116, 1.1051582708008714], [0.48524857309136965, 0.0, 0.0, 0.0, 0.6234636065042042, 0.0], [0.0, 1.311554767551198, 0.0, 2.764457737781053, 0.0, 0.6704216264534533], [0.0, 0.0, 0.11035471549269475, 0.0, 0.0, 0.0]]
    
    # solutions = [[1.0628478285327914, 0.37419878384138777, 0.9908344308914439, 2.0018870370132023, 1.069772550356559, 0.5808659056789868],
    #               [1.0676458646179328, 0.008342508257928007, 1.8312570024824413, 1.8763057997196528, 2.1346039809828654, 0.07650501309898779],
    #               [0.17300315365164587, 2.869052273143852, 0.24676631297870766, 2.0018995273764166, 1.059444020058448, 0.7519263823208308],
    #               [4.45368089332517, 0.005172216856510963, 0.9908370743932583, 0.2109103840244863, 5.225878531131247, 0.5765706867906948],
    #               [1.062868008148948, 0.3740401849486981, 1.8306300284829327, 2.001887034504847, 1.0697745367259848, 0.0770886311973667],
    #               [4.453680893335414, 0.005170035643239698, 1.8312589337734264, 0.21091038375921875, 5.225878576312572, 0.07650439377689613],
    #               [1.0676458646176885, 0.008346028284123713, 0.9908345551992738, 1.8763058098287002, 2.13460396252485, 0.5765782374125282]]
 
    L = [[0.0, 0.0, 0.31825757177906827, 0.0, 0.0, 0.0], [0.14836827215146423, 0.0, 0.0, 0.0, 0.6101806978919241, 0.0], [0.0, 1.2103312970903064, 0.5104685521061533, 0.0, 1.0540821902175936, 0.9926560734861096], [0.15007112473986298, 0.0, 0.0, 0.0, 0.6453862845717143, 0.0], [0.0, 1.0244014370692476, 0.0, 0.7066254271285192, 0.0, 0.6836722027549653], [0.0, 0.0, 0.5328982940180113, 0.0, 0.0, 0.0]]
    U = [[0.0, 0.0, 1.3162653675233926, 0.0, 0.0, 0.0], [0.520034857833214, 0.0, 0.0, 0.0, 2.0366612839061076, 0.0], [0.0, 1.4420940453458586, 0.9652064057450735, 0.0, 2.0562230238104764, 3.917546478338021], [0.6705692455381173, 0.0, 0.0, 0.0, 3.6219311335394635, 0.0], [0.0, 2.11272068738041, 0.0, 3.719994393716693, 0.0, 1.2914336776762338], [0.0, 0.0, 1.2392053213632108, 0.0, 0.0, 0.0]]
    T = [[0.0, 0.0, 0.24978003073504476, 0.0, 0.0, 0.0], [1.3117451495140264, 0.0, 0.0, 0.0, 1.7791313657375478, 0.0], [0.0, 0.6679705891919997, 0.2056155230190702, 0.0, 0.47789230405355976, 0.16627670559691285], [1.3329490533125414, 0.0, 0.0, 0.0, 2.880794068081182, 0.0], [0.0, 5.6984285341889045, 0.0, 1.6757788918054979, 0.0, 0.9503449826558434], [0.0, 0.0, 0.705380046328536, 0.0, 0.0, 0.0]]


    L = np.array(L)
    U = np.array(U)
    T = np.array(T)
    
    D = 6
    # T = g * theta
    theta = T
    gamma = np.ones((D,))
    n = 10

    def F(X, L, U, theta, gamma, n):
        Y = np.zeros(X.shape)
        Y[0] = -gamma[0] * X[0] + ( H_minus(X[1], L[1, 0], U[1, 0], theta[1, 0], n) *
                                    H_minus(X[3], L[3, 0], U[3, 0], theta[3, 0], n) ) # X0 : (~X1) (~X3)
        
        Y[1] = -gamma[1] * X[1] + ( H_minus(X[2], L[2, 1], U[2, 1], theta[2, 1], n) *
                                    H_minus(X[4], L[4, 1], U[4, 1], theta[4, 1], n) ) # X1 : (~X2) (~X4)
        
        Y[2] = -gamma[2] * X[2] + ( H_plus(X[0], L[0, 2], U[0, 2], theta[0, 2], n) *
                                    H_minus(X[2], L[2, 2], U[2, 2], theta[2, 2], n) *
                                    H_minus(X[5], L[5, 2], U[5, 2], theta[5, 2], n) ) # X2 : X0 (~X2) (~X5)
        Y[3] = -gamma[3] * X[3] +   H_minus(X[4], L[4, 3], U[4, 3], theta[4, 3], n)   # X3 : ~X4
        Y[4] = -gamma[4] * X[4] + ( H_plus(X[2], L[2, 4], U[2, 4], theta[2, 4], n) *
                                    H_minus(X[1], L[1, 4], U[1, 4], theta[1, 4], n) *
                                    H_minus(X[3], L[3, 4], U[3, 4], theta[3, 4], n) ) # X4 : X2 (~X1) (~X3)
        Y[5] = -gamma[5] * X[5] + ( H_minus(X[2], L[2, 5], U[2, 5], theta[2, 5], n) *
                                    H_minus(X[4], L[4, 5], U[4, 5], theta[4, 5], n) ) # X5 : (~X2) (~X4)
        return Y
    
    def DH_minus(x, L, U, theta, n):
        return - (U - L) * (n * theta**n * x**(n - 1)) / (x**n + theta**n)**2
    
    def DH_plus(x, L, U, theta, n):
        return (U - L) * (n * theta**n * x**(n - 1)) / (x**n + theta**n)**2
    
    def DF(X, L, U, theta, gamma, n):
        # DF0
        dF0_dX0 = -gamma[0]
        dF0_dX1 = DH_minus(X[1], L[1, 0], U[1, 0], theta[1, 0], n) * H_minus(X[3], L[3, 0], U[3, 0], theta[3, 0], n)
        dF0_dX3 = H_minus(X[1], L[1, 0], U[1, 0], theta[1, 0], n) * DH_minus(X[3], L[3, 0], U[3, 0], theta[3, 0], n)
        DF0 = [dF0_dX0, dF0_dX1, 0, dF0_dX3, 0, 0]
        # DF1
        dF1_dX1 = -gamma[1]
        dF1_dX2 = DH_minus(X[2], L[2, 1], U[2, 1], theta[2, 1], n) * H_minus(X[4], L[4, 1], U[4, 1], theta[4, 1], n)
        dF1_dX4 = H_minus(X[2], L[2, 1], U[2, 1], theta[2, 1], n) * DH_minus(X[4], L[4, 1], U[4, 1], theta[4, 1], n)
        DF1 = [0, dF1_dX1, dF1_dX2, 0, dF1_dX4, 0]
        # DF2
        dF2_dX0 = DH_plus(X[0], L[0, 2], U[0, 2], theta[0, 2], n) * H_minus(X[2], L[2, 2], U[2, 2], theta[2, 2], n)
        dF2_dX2 = -gamma[2] + H_plus(X[0], L[0, 2], U[0, 2], theta[0, 2], n) * DH_minus(X[2], L[2, 2], U[2, 2], theta[2, 2], n)
        DF2 = [dF2_dX0, 0, dF2_dX2, 0, 0, 0]
        # DF3
        dF3_dX3 = -gamma[3]
        dF3_dX4 = DH_minus(X[4], L[4, 3], U[4, 3], theta[4, 3], n)
        DF3 = [0, 0, 0, dF3_dX3, dF3_dX4, 0]
        # DF4
        dF4_dX1 = ( H_plus(X[2], L[2, 4], U[2, 4], theta[2, 4], n) * DH_minus(X[1], L[1, 4], U[1, 4], theta[1, 4], n) *
                  H_minus(X[3], L[3, 4], U[3, 4], theta[3, 4], n) )
        dF4_dX2 = ( DH_plus(X[2], L[2, 4], U[2, 4], theta[2, 4], n) * H_minus(X[1], L[1, 4], U[1, 4], theta[1, 4], n) *
                  H_minus(X[3], L[3, 4], U[3, 4], theta[3, 4], n) )
        dF4_dX3 = ( H_plus(X[2], L[2, 4], U[2, 4], theta[2, 4], n) * H_minus(X[1], L[1, 4], U[1, 4], theta[1, 4], n) *
                  DH_minus(X[3], L[3, 4], U[3, 4], theta[3, 4], n) )  
        dF4_dX4 = -gamma[4]
        DF4 = [0, dF4_dX1, dF4_dX2, dF4_dX3, dF4_dX4, 0]
        # DF5
        dF5_dX2 = DH_minus(X[2], L[2, 5], U[2, 5], theta[2, 5], n) * H_minus(X[4], L[4, 5], U[4, 5], theta[4, 5], n)
        dF5_dX4 = H_minus(X[2], L[2, 5], U[2, 5], theta[2, 5], n) * DH_minus(X[4], L[4, 5], U[4, 5], theta[4, 5], n)
        dF5_dX5 = -gamma[5]
        DF5 = [0, 0, dF5_dX2, 0, dF5_dX4, dF5_dX5]
        # DF
        DF = [DF0, DF1, DF2, DF3, DF4, DF5]
        return DF


    dydt = F(X, L, U, theta, gamma, n)

    return dydt


domain_periodic3d = ((-20,20),(-20,20),(-20,20))
#domain_periodic3d = ((-2,2),(-2,2),(-2,2))
def Periodic_3d(X, t):
    L = [[0.1332269846623764, 1.7430124609497737, 0.0], [0.5995595530787972, 0.12369189840250525, 0.2448043762131514], [0.17473000899364613, 0.0, 0.45830550865929304]]
    U = [[0.5778954292193308, 3.2995532202573132, 0.0], [1.9945763853522422, 0.7254306497031602, 5.205353010194985], [1.7368185795303968, 0.0, 2.1639849528341295]]
    T = [[0.3629413877601428, 1.5308315346787003, 0.0], [2.0362303064213543, 0.8624713186039009, 0.23355676586045557], [0.8182265975281928, 0.0, 0.1684576070743367]]
    # Define theta, gamma, and n
    L = np.array(L)
    U = np.array(U)
    T = np.array(T)
    
    D = 3
    
    # T = gamma * theta
    theta = T
    gamma = np.ones((D,))
    # theta = T
    # gamma = np.ones((D,))
    n = 10
    # Define Hill model for network
    def H_minus(x, L, U, theta, n):
        return L + (U - L) * theta**n / (x**n + theta**n)
    
    def H_plus(x, L, U, theta, n):
        return L + (U - L) * x**n / (x**n + theta**n)
    
    def F(X, L, U, theta, gamma, n):
        Y = np.zeros(X.shape)
        Y[0] = -gamma[0] * X[0] + ( H_plus(X[0], L[0, 0], U[0, 0], theta[0, 0], n) *
                                    H_minus(X[1], L[1, 0], U[1, 0], theta[1, 0], n) *
                                    H_minus(X[2], L[2, 0], U[2, 0], theta[2, 0], n) ) # x : x (~y) (~z)
        
        Y[1] = -gamma[1] * X[1] + ( H_plus(X[1], L[1, 1], U[1, 1], theta[1, 1], n) *
                                    H_minus(X[0], L[0, 1], U[0, 1], theta[0, 1], n) ) # y : y (~x)
        
        Y[2] = -gamma[2] * X[2] + ( H_plus(X[2], L[2, 2], U[2, 2], theta[2, 2], n) *
                                    H_minus(X[1], L[1, 2], U[1, 2], theta[1, 2], n) ) # z : z (~y)
        return Y
    
    # Map f
    def f(X):
        return F(X, L, U, theta, gamma, n)
    
    dydt = F(X, L, U, theta, gamma, n)
    return dydt

domain_memristive = ((-1.6,1.6),(-2.2,2.2),(-1.2,1.2))
def Memristive(y, t):
    a = 0.5
    b = 0.8
    c = 0.6
    d = 7
    k = 1.0
    s = 0.02
    r = (y[0]**2 - s*y[0] - 2) * y[1]
    y0 = y[1]
    y1 = d * y[2]
    y2 = -a * y[2] + b * y[0] - c * y[0]**3 + k * r
    dydt = np.array([y0,y1,y2])
    return dydt



def plot2dv(pts, domain, labeled_pts):
    # Plot labeled dataset
    plt.figure()
    plt.clf()
    colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
    
    num_of_labels = int(max(labeled_pts[:,-1]) + 1)

    for k, col in zip(range(num_of_labels), colors):
        my_members = labeled_pts[:,-1] == k
        #cluster_center = ai.cluster_centers[k]
        plt.plot(pts[my_members, 0], pts[my_members, 1], col + ".")

    plt.title(f"{num_of_pts} Points")
    #domain = ((0,7),(0.5,6))
    plt.xlim((domain[0][0], domain[0][1]))
    plt.ylim((domain[1][0], domain[1][1]))
    plt.show()
    

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
          
def plot3d(pts, domain, labels, set_zlim=False, title="Points"):
    # Plot lifted dataset
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #plt.figure()
    #ax = plt.axes(projection='3d')
    #colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
    num_of_labels = int(max(labels) + 1)
    
    color_map = plt.get_cmap('Paired', num_of_labels)
    colors = [color_map(i) for i in range(num_of_labels)]

    for k, col in zip(range(num_of_labels), colors):
        my_members = labels == k

        ax.scatter3D(pts[my_members,0], pts[my_members,1], pts[my_members,2], color=col)

    if set_zlim:
        ax.set_zlim(domain[2][0], domain[2][1])  
    plt.title(title)

    plt.show()        
    return 

def plotVF(DS, domain, radial):
    plt.figure()
    grid = make_grid(2, 10, domain)  
    t=1
    dydt = np.array([DS(g, t)/np.linalg.norm(DS(g, t)) for g in grid])
    # # Meshgrid
    eps = 0.5
    x, y = np.meshgrid(np.linspace(domain[0][0]+eps, domain[0][1]+eps, 10), 
                       np.linspace(domain[1][0]+eps, domain[1][1]+eps, 10))

      
    # Plotting Vector Field with QUIVER
    if radial:
        plt.quiver(grid[:,0], grid[:,1], dydt[:,0]*np.cos(dydt[:,1]), dydt[:,0]*np.sin(dydt[:,1]), color='b')
    else:
        plt.quiver(grid[:,0], grid[:,1], dydt[:,0], dydt[:,1], color='b')
        
    plt.title('Vector Field')
      
    # Setting x, y boundary limits
    plt.xlim(domain[0][0]+eps, domain[0][1]+eps)
    plt.ylim(domain[1][0]+eps, domain[1][1]+eps)
      
    # Show plot with grid
    plt.grid()
    plt.show()

if __name__=='__main__':   

        systems = {
            Straight : domain_straight, 
            Curved : domain_curved, 
            Periodic : domain_periodic, 
            Radial : domain_radial, 
            Ellipsoidal : domain_ellipsoidal,
            Ellipsoidal_3d : domain_ellipsoidal3d,
            Periodic_3d : domain_periodic3d,
            Straight_4d : domain_straight4d,
            EMT : domain_emt,
            Duffing : domain_duffing,
            Harmonic : domain_harmonic,
            Memristive : domain_memristive


        }
        
        
        DS = Memristive
        num_of_pts = 100


        grid = False
        domain = systems[DS]
        dim = len(domain)
        step_size = 1 #2 * np.pi #1
        eps = 1e-6
        max_iter = 100
      
        if DS == Radial or DS == Ellipsoidal or DS == Ellipsoidal_3d:
            radial = True
        else:
            radial = False
            
        X0 = init_pts(domain, num_of_pts) 
        X1, hausdorf_distances = remove_transience(DS, X0, step_size, max_iter, eps, domain, radial)  
        s0 = compute_norms(X1)

    

        M = 100 * len(hausdorf_distances)         
        X1, s = iter_and_compute_norms(DS, X1, M, step_size, domain, radial)
        
        
        index_pts_in_domain = np.where(np.linalg.norm(X1[-1], axis=1)<=max(np.linalg.norm(domain, axis=1))*1.5)
        index_pts_out_domain = np.where(np.linalg.norm(X1[-1], axis=1)>max(np.linalg.norm(domain, axis=1))*1.5)


        lifted_pts_in_domain = np.hstack([X1[-1,:,:], s])[index_pts_in_domain]
        lifted_pts_out_domain = np.hstack([X1[-1,:,:], s-s-1])[index_pts_out_domain]   
        #lifted_pts = np.vstack((lifted_pts_in_domain,lifted_pts_out_domain))   
 
        

        if dim==2:
            plotVF(DS, domain, radial)
            num_points_per_dim = 100

        if dim==3:
            num_points_per_dim = 25
            
        plt.figure()
        plt.plot(hausdorf_distances)
        plt.title(
            f"Hausdorff Distance Between Iterations")
        
        diag = compute_persistance(lifted_pts_in_domain)
        gudhi.plot_persistence_diagram(diag)
        plt.show()
        

        
        print(diag[:10])
        resolution = float(input("What is the resolution? ")) 
        labels = get_labels(lifted_pts_in_domain, resolution)
        
        
        labeled_pts_in_domain = np.hstack((X0[index_pts_in_domain],np.expand_dims(labels, axis=1)))
        labeled_pts_out_domain = np.hstack((X0[index_pts_out_domain],-1*np.ones((len(index_pts_out_domain[0]),1))))
        labeled_pts = np.vstack((labeled_pts_in_domain,labeled_pts_out_domain))   
        
        plot2d(labeled_pts_in_domain, domain, labels, title="Initial Points with Attractor Color")
        
        plot2d(lifted_pts_in_domain, domain, labels, title="Final Points with Attractor Color", keep_scale=False)
     
        plot3d(lifted_pts_in_domain, domain, labels, title="Final Points with Attractor Color")
        
        plot2d(lifted_pts_in_domain, domain, lifted_pts_in_domain[:,-1], title="Final Points with Attractor Color", keep_scale=False)

                      

        
        #np.savetxt("Curved_6d_labeled_set2.csv", labeled_pts)
        #np.savetxt("Periodic3d_lifted_set.csv", lifted_pts)
 
        
        if DS == Ellipsoidal:
            labeled_pts_in_domain2 = np.copy(labeled_pts_in_domain)
            labeled_pts_in_domain2[:,0] *= 2
            R = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)], [np.sin(np.pi/4), np.cos(np.pi/4)]])
            labeled_pts_in_domain3 = np.hstack([(R @ labeled_pts_in_domain2[:,0:2].T).T, labels.reshape(-1,1)])        
            labeled_pts_in_domain4 = np.array([labeled_pts for labeled_pts in labeled_pts_in_domain3 if np.abs(labeled_pts[0])<4 and np.abs(labeled_pts[1])<4])
            
            plot2d(labeled_pts_in_domain4[0:10000], domain, labeled_pts_in_domain4[:,-1][0:10000])     
            
            

        
        
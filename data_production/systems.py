#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 15:46:20 2024

@author: paultatasciore
"""



import numpy as np
#import NN_class
#import load_data


# system 1: Two stable fixed points (straight separatrix) in 2-d
domain_straight = ((-2,2),(-3.5,3.5))
def Straight(y, t):
    # Kalies Simple Example
    theta=np.pi/3
    u=[np.cos(theta)*y[0]+np.sin(theta)*y[1], -np.sin(theta)*y[0]+np.cos(theta)*y[1]]
    ud=u[0]*(1-u[0]**2)
    vd=(u[0]**2)*(3-2*u[0]**2)-u[1]        
    dydt=[np.cos(theta)*ud-np.sin(theta)*vd, np.sin(theta)*ud+np.cos(theta)*vd]
    return dydt

# system 2: Two stable fixed points (curved separatrix) in 2-d
domain_curved = ((-2,2),(-3.5,3.5))    
def Curved(y, t):   
      p=1
      e=3   
      ud = -y[0]
      vd = (y[1] - p * y[0]**2) * (e**2 - y[1]**2)
      dydt = [ud, vd]
      return dydt

# system 3: Radial system in 2-d with two labels
domain_radial2 = ((-4,4),(-4,4))
def Radial2(y, t):
    # Radial System
    theta = 1
    r = - y[0] * (y[0]-1) * (y[0]-2) * (y[0]-3) 
    dydt = np.array([r,theta])
    return dydt

# system 4: Radial system in 2-d with three labels
domain_radial = ((-5,5),(-5,5))
def Radial(y, t):
    # Radial System
    theta = 1
    r = - y[0] * (y[0]-1) * (y[0]-2) * (y[0]-3) * (y[0]-4) 
    dydt = np.array([r,theta])
    return dydt

# system 5: Two stable fixed points (straight separatrix) in 4-d
domain_straight4d = ((-2,2),(-3.5,3.5),(-2,2),(-2,2))    
def Straight_4d(y, t):    
    theta=np.pi/3
    u=[np.cos(theta)*y[0]+np.sin(theta)*y[1], -np.sin(theta)*y[0]+np.cos(theta)*y[1]]
    ud=u[0]*(1-u[0]**2)
    vd=(u[0]**2)*(3-2*u[0]**2)-u[1]            
    wd = -y[2]
    xd = -y[3]
    dydt=[np.cos(theta)*ud-np.sin(theta)*vd, np.sin(theta)*ud+np.cos(theta)*vd, wd, xd]
    return dydt

# system 6: Two stable fixed points (curved separatrix) in 4-d
domain_curved4d = ((-2,2),(-3.5,3.5),(-2,2),(-2,2))   
def Curved_4d(y, t):   
      p=1
      e=3   
      ud = -y[0]
      vd = (y[1] - p * y[0]**2) * (e**2 - y[1]**2)
      wd = -y[2]
      xd = -y[3]
      dydt = [ud, vd, wd, xd]
      return dydt

# system 7: EMT hill system in 6-d
#domain_emt = ((0,149),(0,323),(0,459),(0,189),(0,397),(0,355))
#domain_emt = (((0,3.75),)*6) 
#e = 5
#domain_emt = ((0.0222807-e,0.0222807+e),(3.04674-e,3.04674+e),(0.158407-e,0.158407+e),(3.71994-e,3.71994+e),(0.558874-e,0.558874+e),(3.6113-e,3.6113+e))
#domain_emt = ((0,1.42),(0,1.6),(0,3.8),(0,0.01),(0,1.7),(0,1))

domain_emt = ((0,0.5),(0,3.558),(0,1.5734556600000003),(0,5.762),(0,15.169196784),(0,5.056846999999999))

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

    # # Number stable equilibria: 2 (found) 
    # L = [[0.0, 0.0, 0.31825757177906827, 0.0, 0.0, 0.0], [0.14836827215146423, 0.0, 0.0, 0.0, 0.6101806978919241, 0.0], [0.0, 1.2103312970903064, 0.5104685521061533, 0.0, 1.0540821902175936, 0.9926560734861096], [0.15007112473986298, 0.0, 0.0, 0.0, 0.6453862845717143, 0.0], [0.0, 1.0244014370692476, 0.0, 0.7066254271285192, 0.0, 0.6836722027549653], [0.0, 0.0, 0.5328982940180113, 0.0, 0.0, 0.0]]
    # U = [[0.0, 0.0, 1.3162653675233926, 0.0, 0.0, 0.0], [0.520034857833214, 0.0, 0.0, 0.0, 2.0366612839061076, 0.0], [0.0, 1.4420940453458586, 0.9652064057450735, 0.0, 2.0562230238104764, 3.917546478338021], [0.6705692455381173, 0.0, 0.0, 0.0, 3.6219311335394635, 0.0], [0.0, 2.11272068738041, 0.0, 3.719994393716693, 0.0, 1.2914336776762338], [0.0, 0.0, 1.2392053213632108, 0.0, 0.0, 0.0]]
    # T = [[0.0, 0.0, 0.24978003073504476, 0.0, 0.0, 0.0], [1.3117451495140264, 0.0, 0.0, 0.0, 1.7791313657375478, 0.0], [0.0, 0.6679705891919997, 0.2056155230190702, 0.0, 0.47789230405355976, 0.16627670559691285], [1.3329490533125414, 0.0, 0.0, 0.0, 2.880794068081182, 0.0], [0.0, 5.6984285341889045, 0.0, 1.6757788918054979, 0.0, 0.9503449826558434], [0.0, 0.0, 0.705380046328536, 0.0, 0.0, 0.0]]
        

    # L = np.array(L)
    # U = np.array(U)
    # T = np.array(T)
    
    # D = 6
    # # T = g * theta
    # theta = T
    # gamma = np.ones((D,))
    
    # EMT 6D System
    L = [[0.0, 0.0, 0.318, 0.0, 0.0, 0.0], [0.148, 0.0, 0.0, 0.0, 0.610, 0.0], [0.0, 1.210, 0.510, 0.0, 1.054, 0.993], [0.150, 0.0, 0.0, 0.0, 0.6454, 0.0], [0.0, 1.024, 0.0, 0.707, 0.0, 0.684], [0.0, 0.0, 0.533, 0.0, 0.0, 0.0]]
    U = [[0.0, 0.0, 1.316, 0.0, 0.0, 0.0], [0.520, 0.0, 0.0, 0.0, 2.037, 0.0], [0.0, 1.442, 0.965, 0.0, 2.056, 3.917], [0.670, 0.0, 0.0, 0.0, 3.622, 0.0], [0.0, 2.113, 0.0, 3.720, 0.0, 1.291], [0.0, 0.0, 1.239, 0.0, 0.0, 0.0]]
    T = [[0.0, 0.0, 0.250, 0.0, 0.0, 0.0], [1.312, 0.0, 0.0, 0.0, 1.779, 0.0], [0.0, 0.668, 0.206, 0.0, 0.478, 0.166], [1.333, 0.0, 0.0, 0.0, 2.881, 0.0], [0.0, 5.698, 0.0, 1.676, 0.0, 0.950], [0.0, 0.0, 0.705, 0.0, 0.0, 0.0]]
    
    L = np.array(L)
    U = np.array(U)
    T = np.array(T)
    
    D = 6
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

# system 8: Periodic system in 3-d
#domain_periodic3d = ((-20,20),(-20,20),(-20,20))
#domain_periodic3d = ((0,2),(0,3.3),(0,5.3))
#domain_periodic3d = ((-2,2),(-2,2),(-2,2))
#domain_periodic3d = ((0,2.5),(0,2.5),(0,2.5))
domain_periodic3d = ((0,3.062),(0,4.072),(0,11.263620000000001))


def Periodic_3d(X, t):
    # L = [[0.1332269846623764, 1.7430124609497737, 0.0], [0.5995595530787972, 0.12369189840250525, 0.2448043762131514], [0.17473000899364613, 0.0, 0.45830550865929304]]
    # U = [[0.5778954292193308, 3.2995532202573132, 0.0], [1.9945763853522422, 0.7254306497031602, 5.205353010194985], [1.7368185795303968, 0.0, 2.1639849528341295]]
    # T = [[0.3629413877601428, 1.5308315346787003, 0.0], [2.0362303064213543, 0.8624713186039009, 0.23355676586045557], [0.8182265975281928, 0.0, 0.1684576070743367]]
    # # Define theta, gamma, and n
    # L = np.array(L)
    # U = np.array(U)
    # T = np.array(T)
        
    # D = 3
 
    # # T = gamma * theta
    # theta = T
    # gamma = np.ones((D,))
    
    # Periodic3D System
    # 1 periodic orbit, 3 fixed points
    L = [[0.133, 1.743, 0.0], [0.599, 0.124, 0.245], [0.175, 0.0, 0.458]]
    U = [[0.578, 3.299, 0.0], [1.994, 0.725, 5.205], [1.737, 0.0, 2.164]]
    T = [[0.363, 1.531, 0.0], [2.036, 0.862, 0.233], [0.818, 0.0, 0.168]]
    
    # Define theta, gamma, and n
    L = np.array(L)
    U = np.array(U)
    T = np.array(T)
    
    D = 3
    theta = T
    gamma = np.ones((D,))

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

# system 9: Ellipsoidal system in 2-d

domain_ellipsoidal = ((-4,4),(-4,4))
def Ellipsoidal(y, t):
    # Ellipsoidal System

    r = np.linalg.norm(y)
    r_dot = - r * (r-1) * (r-2) * (r-3) 

    x_term = y[0]/r
    y_term = y[1]/r

    x0 = r_dot*x_term - y[1]
    x1 = r_dot*y_term + y[0]
    
    dydt = np.array([x0,x1])
    
    return dydt

domain_periodic = ((0,7),(0.5,6))
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

domain_periodic_back = ((0,7),(0.5,6))
def Periodic_back(y,t):
    r = 1
    K = 10
    m = 0.54
    a = 1.25
    b = -1.65
    s = 0.4
    y0 = r * y[0] * (1 - y[0] / K) - (m * y[1] * y[0]**2) /(a * y[0]**2 + b * y[0] + 1)
    y1 = s * y[1] * (1 - y[1] / y[0])
    dydt = np.array([y0, y1])
    return -dydt

domain_torus = ((-2.5,2.5), (-0.2,0.2), (-0.3,0.3), (-1.4,0.1), (-10,10))
#domain_torus = ((-1.5,1.5), (-1.5,1.5), (-1.5,1.5), (-1.5,1.5), (-1.5,1.5))

def Torus(y,t):
    # Marcio Example 1   
    rho = 1.2e-6
    phi = 1 / 0.026
    c = 0.56 # 1.55

    y0 = y[1] - rho * np.sinh(phi * y[2])
    y1 = y[2] - y[1]
    y2 = y[3]
    y3 = y[4]
    y4 = -c * y[4] - rho * np.sinh(phi * y[3]) - 5 * y[2] - 5 * y[1] - 0.1 * y[0]             
    dydt = np.array([y0, y1, y2, y3, y4])
    return dydt

domain_leslie = ((0, 74), (0, 52))
def Leslie(x, dim):   
    f = 20
    param = {'lam': 0.1, 'th1': f, 'th2': f, 'p1': 0.7}

    # 2d Leslie    
    gx = np.array([(param['th1'] * x[:,0] + param['th2'] * x[:,1]) * np.exp(-param['lam']*(x[:,0] + x[:,1])), param['p1']*x[:,0]])
    return gx.T

domain_iris = ((-1, 1),)*11
def Iris(x, dim, sys=1): 
    param = {'bias': True}


    def f_NN(X):
        Neural = NN_class.NN(hidden_layer_size=1, activation='relu', learning_rate=1, max_iter=1, bias=param['bias'])
        acc, err = Neural.train_test(load_data.x, load_data.y, initial='custom', init=X[0])
        weights = Neural.final_weights
        final = np.append(weights[0][:],weights[1][:])
        if param['bias']:
            final = np.append(final,weights[2][:])
            final = np.append(final,weights[3][:])
        # final = np.append(final,acc)
        # final = np.append(final,err)
        final = np.expand_dims(final, axis=0)
        # Final = np.array([])
        # for i in range(len(X)):
        #     Final = np.append(Final, f(np.expand_dims(X[i], axis=0)))
        #     print(i)       
        # X = X[0:i+1,:]
        # Final = np.reshape(Final, (i+1,13))
        return final, acc
               
    def F(Initial,dim):
        Final = np.array([])
        Acc = np.array([])
        for i in range(len(Initial)):
            Final = np.append(Final, f_NN(np.expand_dims(Initial[i], axis=0))[0])
            Acc = np.append(Acc, f_NN(np.expand_dims(Initial[i], axis=0))[1])
            #print(i)       
        #Initial = Initial[0:i+1,:]
        Final = np.reshape(Final, (i+1,dim))
        return Final, Acc 
    gx, Acc = F(x,dim)
    print(Acc)
    return gx, Acc
# # system 9: Ellipsoidal system in 2-d
# domain_ellipsoidal = ((-4,4),(-4,4))
# def Ellipsoidal(y, t):
#     # Ellipsoidal System
    
#     r = np.linalg.norm(y)
#     r_dot = - r * (r-1) * (r-2) * (r-3) #np.sqrt(a**2 * np.sin(y[1])**2 + b**2 * np.cos(y[1])**2)

#     x_term = y[0]/r
#     y_term = y[1]/r

#     x0 = r_dot*x_term - y[1]
#     x1 = r_dot*y_term + y[0]
    
#     dydt = np.array([x0,x1])
    
#     #dydt = np.array([np.sqrt(dydt[0]**2 + dydt[1]**2), np.arctan2(dydt[1],dydt[0])])
    
#     #np.array([[[y[0] * np.cos(y[1]), y[0] * np.sin(y[1])]
    
#     return dydt

domain_lorentz = ((-20,20),(-20,20),(-20,20))
def Lorentz(y, t):
    sigma = 10
    beta = 8/3
    r=12 #12 20

    
    x0 = sigma * (y[1] - y[0])
    x1 = r*y[0] - y[1] - y[0]*y[2]
    x2 = y[0]*y[1] - beta*y[2]
    
    
    dydt = np.array([x0,x1,x2])
    
    return dydt

domain_fp3 = ((0,3.5),(0,5.5))
def FP3(y, t):
    L = [[0.771, 1.331], [0.215, 0.223]]
    U = [[2.124, 1.585], [1.064, 1.823]]
    T = [[0.925, 0.523], [2.638, 0.848]]

    # Define theta, gamma, and n
    L = np.array(L)
    U = np.array(U)
    T = np.array(T)
    
    D = 2
    
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
                                    H_minus(X[1], L[1, 0], U[1, 0], theta[1, 0], n) ) # x : x (~y)
        Y[1] = -gamma[1] * X[1] + ( H_plus(X[1], L[1, 1], U[1, 1], theta[1, 1], n) *
                                    H_minus(X[0], L[0, 1], U[0, 1], theta[0, 1], n) ) # y : y (~x)
        return Y
    
    # Map f
    def f(X):
        return F(X, L, U, theta, gamma, n)
    dydt = F(y, L, U, theta, gamma, n)
    
    return dydt


domain_inf = ((0.0,0.3),(-0.5,.5))    
def Inf(y, t):     
      xd = y[0] * np.sin(1.0/y[0])
      yd = -y[1]
      dydt = [xd, yd]
      return dydt

domain_memristive = ((-0.6,0.6),(-0.2,0.2),(-0.2,0.2))
def Memristive(y, t):
    a = 0.5
    b = 0.8
    c = 0.6
    d = 7.0
    k = 1.0
    s = 0.0
    r = (y[0]**2 - s*y[0] - 2) * y[1]
    y0 = y[1]
    y1 = d * y[2]
    y2 = -a * y[2] + b * y[0] - c * y[0]**3 + k * r
    dydt = np.array([y0,y1,y2])
    return dydt

domain_delay = ((-4,4),(0,5))
def Delay(y, t):

    y0 = 2*y[0] - 1.1*y[0]*y[1]
    y1 = -y[1] + 0.9*y[0]*y[1]
    dydt = np.array([y0,y1])
    return dydt


systems = {
    Straight : domain_straight, 
    Curved : domain_curved, 
    Radial : domain_radial, 
    Radial2 : domain_radial2, 
    Straight_4d : domain_straight4d,
    Curved_4d : domain_straight4d,
    EMT : domain_emt,
    Periodic_3d : domain_periodic3d, 
    Ellipsoidal : domain_ellipsoidal,
    Periodic : domain_periodic,
    Periodic_back : domain_periodic_back,
    Torus : domain_torus,
    Leslie : domain_leslie,
    Iris : domain_iris,
    Lorentz : domain_lorentz,
    FP3 : domain_fp3,
    Inf : domain_inf,
    Memristive : domain_memristive,
    Delay : domain_delay


}
   
        

       


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 19:26:30 2024

@author: paultatasciore
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

def compute_cube_labels(data_on_grid, N, dim):
    l=data_on_grid[:,dim].astype(int)
    l=l.reshape([N]*dim)
    cube_labels=np.empty([N-1]*dim)
    if dim==2:
        for i in range(N-1):
            for j in range(N-1):
                s=[l[i,j], l[i+1,j], l[i+1,j+1], l[i,j+1]]
                if min(s)==max(s):
                    cube_labels[i,j]=min(s)
                else:
                    cube_labels[i,j]=-1           
                    
    if dim==3:
        
        count=[0,0,0,0,0,0]
        for i0 in range(N-1):
            for i1 in range(N-1):
                for i2 in range(N-1):
                    s=[]
                    I=[i0,i1,i2]
                    for j in range(8):
                        P = [int(i) for i in list('{0:03b}'.format(j))]
                        J=[I[i]+P[i] for i in range(3)]
                        s.append(l[J[0],J[1],J[2]])
                    if min(s)==max(s):
                        cube_labels[i0,i1,i2]=min(s)
                        count[min(s)]=count[min(s)]+1
                    else:
                        cube_labels[i0,i1,i2]=-1        
                        count[4]=count[4]+1                            
        print("Box counts for each attractor and the separatrix")
        print(count)
        print("% of total boxes each attractor and the separatrix")
        print(100*np.array(count)/1000000)     

    if dim ==4:
        count=[0,0,0,0,0,0,0]
        for i0 in range(N-1):
            for i1 in range(N-1):
                for i2 in range(N-1):
                    for i3 in range(N-1):
                        s=[]
                        I=[i0,i1,i2,i3]
                        for j in range(16):
                            P = [int(i) for i in list('{0:04b}'.format(j))]
                            J=[I[i]+P[i] for i in range(4)]
                            s.append(l[J[0],J[1],J[2],J[3]])
                        if min(s)==max(s):
                            cube_labels[i0,i1,i2,i3]=min(s)
                            count[min(s)]=count[min(s)]+1
                        else:
                            cube_labels[i0,i1,i2,i3]=-1        
                            count[5]=count[5]+1                            
        print("Box counts for each attractor and the separatrix")
        print(count) 
        
    if dim == 6:
        count=[0,0,0,0,0,0,0,0,0]
        for i0 in range(N-1):
            for i1 in range(N-1):
                for i2 in range(N-1):
                    for i3 in range(N-1):
                        for i4 in range(N-1):
                            for i5 in range(N-1):
                                s=[]
                                I=[i0,i1,i2,i3,i4,i5]
                                for j in range(64):
                                    P = [int(i) for i in list('{0:06b}'.format(j))]
                                    J=[I[i]+P[i] for i in range(6)]
                                    s.append(l[J[0],J[1],J[2],J[3],J[4],J[5]])
                                if min(s)==max(s):
                                    cube_labels[i0,i1,i2,i3,i4,i5]=min(s)
                                    count[min(s)]=count[min(s)]+1
                                else:
                                    cube_labels[i0,i1,i2,i3,i4,i5]=-1        
                                    count[8]=count[8]+1                            
        print("Box counts for each attractor and the separatrix")
        print(count)
        print("% of total boxes each attractor and the separatrix")
        print(100*np.array(count)/531441)
    
    return cube_labels



def plot_hom(cube_labels, domain, N, X):
    dim = len(domain)
    if dim==2:
        fig, ax=plt.subplots();
        ax.set(xlim=(domain[0][0],domain[0][1]),ylim=(domain[1][0],domain[1][1]))
        fig.set_size_inches(6,6)    
    
        dx=X[0][1]-X[0][0]
        dy=X[1][1]-X[1][0]
    
        a0_count=0
        a1_count=0
        a2_count=0
        s_count=0
    
        A0=[]
        A1=[]
        A2=[]
        S=[]
    
        for i in range(N-1):
            for j in range(N-1):
                rect=Rectangle((X[0][i],X[1][j]),dx,dy);
                if(cube_labels[i,j]==0):           
                    A0.append(rect)
                    a0_count=a0_count+1
                if(cube_labels[i,j]==1):           
                    A1.append(rect)
                    a1_count=a1_count+1
                if(cube_labels[i,j]==2):           
                    A2.append(rect)
                    a2_count=a2_count+1
                if(cube_labels[i,j]==-1):           
                    S.append(rect)
                    s_count=s_count+1
    
        ap0=PatchCollection(A0,facecolor='b',edgecolor='none',alpha=1) #alpha=0.5) 
        ap1=PatchCollection(A1,facecolor='c',edgecolor='none',alpha=1) #alpha=0.5) 
        ap2=PatchCollection(A2,facecolor='g',edgecolor='none',alpha=1) #alpha=0.5) 
        sp=PatchCollection(S,facecolor='k',edgecolor='none',alpha=0.5) #alpha=0.5) 
    
        ax.add_collection(ap0)    
        ax.add_collection(ap1)  
        ax.add_collection(ap2)  
        ax.add_collection(sp) 
        
    
        print([a0_count,a1_count,a2_count,s_count])

    elif dim==3:    

        def cube3d(ll,dx,dy,dz):
            cubes = [
            [(ll[0],ll[1],ll[2]), (ll[0],ll[1]+dy,ll[2]), (ll[0]+dx,ll[1]+dy,ll[2]), (ll[0]+dx,ll[1],ll[2])],  # Cube 1 (bottom face)
            [(ll[0],ll[1],ll[2]+dz), (ll[0],ll[1]+dy,ll[2]+dz), (ll[0]+dx,ll[1]+dy,ll[2]+dz), (ll[0]+dx,ll[1],ll[2]+dz)],  # Cube 1 (top face)
            [(ll[0]+dx,ll[1],ll[2]), (ll[0]+dx,ll[1]+dy,ll[2]), (ll[0]+dx,ll[1]+dy,ll[2]+dz), (ll[0]+dx,ll[1],ll[2]+dz)],  # Cube 2 (right face)
            [(ll[0],ll[1],ll[2]), (ll[0],ll[1]+dy,ll[2]), (ll[0],ll[1]+dy,ll[2]+dz), (ll[0],ll[1],ll[2]+dz)],  # Cube 3 (left face)
            [(ll[0],ll[1],ll[2]), (ll[0]+dx,ll[1],ll[2]), (ll[0]+dx,ll[1],ll[2]+dz), (ll[0],ll[1],ll[2]+dz)],  # Cube 4 (front face)
            [(ll[0],ll[1]+dy,ll[2]), (ll[0]+dx,ll[1]+dy,ll[2]), (ll[0]+dx,ll[1]+dy,ll[2]+dz), (ll[0],ll[1]+dy,ll[2]+dz)],  # Cube 5 (back face)
            ]
            return(cubes)
        # Create a figure and a 3D axis
        def PlotRegion(R):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            x=X[0]
            y=X[1]
            z=X[2]
            
            dx=x[1]-x[0]
            dy=y[1]-y[0]
            dz=z[1]-z[0]
        
            a0_count=0
            a1_count=0
            a2_count=0
            a3_count=0
            s_count=0
            
            #A0=[]
            #A1=[]
            #A2=[]
            #A3=[]
            #S=[]
            
            X0=[]
            Y0=[]
            Z0=[]
            
            X1=[]
            Y1=[]
            Z1=[]
            
            X2=[]
            Y2=[]
            Z2=[]
            
            X3=[]
            Y3=[]
            Z3=[]
            
            XS=[]
            YS=[]
            ZS=[]
        
            for i in range(N-1):
                for j in range(N-1):
                    for k in range(N-1):
                        #rect=cube3d([x[i],y[j],z[k]],dx,dy,dz)
                        #if ((i==0)&(j==0)&(k==0)):
                        #    print(rect)
                        if(cube_labels[i,j,k]==0):           
                            #A0.append(rect)
                            #A0=A0+rect
                            X0.append(x[i]+dx/2)
                            Y0.append(y[j]+dy/2)
                            Z0.append(z[k]+dz/2)
                            a0_count=a0_count+1
                        if(cube_labels[i,j,k]==1):  
                            #A1.append(rect)
                            #A1=A1+rect
                            X1.append(x[i]+dx/2)
                            Y1.append(y[j]+dy/2)
                            Z1.append(z[k]+dz/2)
                            a1_count=a1_count+1
                        if(cube_labels[i,j,k]==2):           
                            #A2.append(rect)
                            #A2=A2+rect
                            X2.append(x[i]+dx/2)
                            Y2.append(y[j]+dy/2)
                            Z2.append(z[k]+dz/2)
                            a2_count=a2_count+1  
                        if(cube_labels[i,j,k]==3):           
                            #A3.append(rect)
                            #A3=A3+rect
                            X3.append(x[i]+dx/2)
                            Y3.append(y[j]+dy/2)
                            Z3.append(z[k]+dz/2)
                            a3_count=a3_count+1      
                        if(cube_labels[i,j,k]==-1):           
                            #S.append(rect)
                            #S=S+rect
                            XS.append(x[i]+dx/2)
                            YS.append(y[j]+dy/2)
                            ZS.append(z[k]+dz/2)
                            s_count=s_count+1
        
            if(R==0):
                ax.scatter3D(X0,Y0,Z0,color='b',alpha=0.5)
            if(R==1):    
                ax.scatter3D(X1,Y1,Z1,color='c',alpha=0.5)
            if(R==2):     
                ax.scatter3D(X2,Y2,Z2,color='r',alpha=0.5)
            if(R==3): 
                ax.scatter3D(X3,Y3,Z3,color='m',alpha=0.5)
            if(R==4):     
                ax.scatter3D(XS,YS,ZS,color='k',alpha=0.5)
            if(R==5):  
                #ax.scatter3D(XS,YS,ZS,color='k',alpha=0.5)  
                ax.scatter3D(X0,Y0,Z0,color='b',alpha=0.5)
                ax.scatter3D(X1,Y1,Z1,color='c',alpha=0.5)
                ax.scatter3D(X2,Y2,Z2,color='r',alpha=0.5)
                ax.scatter3D(X3,Y3,Z3,color='m',alpha=0.5)
                #ax.scatter3D(XS,YS,ZS,color='k',alpha=0.5)  
            if(R==6):  
                ax.scatter3D(X0,Y0,Z0,color='b',alpha=0.5)
                ax.scatter3D(X1,Y1,Z1,color='c',alpha=0.5)
                ax.scatter3D(X2,Y2,Z2,color='r',alpha=0.5)
                ax.scatter3D(X3,Y3,Z3,color='m',alpha=0.5)
                ax.scatter3D(XS,YS,ZS,color='k',alpha=0.5)  
        
            # # Create the Poly3DCollection object
            # A0_collection = Poly3DCollection(A0, facecolors='b', alpha=0.5)
            # A1_collection = Poly3DCollection(A1, facecolors='c', alpha=0.5)
            # A2_collection = Poly3DCollection(A2, facecolors='r', alpha=0.5)
            # A3_collection = Poly3DCollection(A3, facecolors='m', alpha=0.5)
            # S_collection = Poly3DCollection(S, facecolors='k', alpha=0.5)
            
            # # Add the collection to the axes
            # ax.add_collection3d(A0_collection)
            # ax.add_collection3d(A1_collection)
            # ax.add_collection3d(A2_collection)
            # ax.add_collection3d(A3_collection)
            # ax.add_collection3d(S_collection)
            
            # Set the axes limits and labels
            ax.set_xlim([domain[0][0],domain[0][1]])
            ax.set_ylim([domain[1][0],domain[1][1]])
            ax.set_zlim([domain[2][0],domain[2][1]])
            
            # Show the plot
            plt.show()
        
        for i in range(7):
            PlotRegion(i)
    else:
        print("No hom plots for dim > 3")
            

def write_cube_labels(cube_labels, dim, N, path_out):
    if dim==2:
        with open(path_out + 'att0.txt','w') as outfile0, open(path_out + 'att1.txt','w') as outfile1, open(path_out + 'att2.txt','w') as outfile2, open(path_out + 'sep.txt','w') as outfile3:              
            for i in range(N-1):
                for j in range(N-1):
                    if cube_labels[i,j]==0:
                        outfile0.write('(%d,%d)\n' % (i,j))
                    elif cube_labels[i,j]==1:
                        outfile1.write('(%d,%d)\n' % (i,j))
                    elif cube_labels[i,j]==2:
                        outfile2.write('(%d,%d)\n' % (i,j))
                    elif cube_labels[i,j]==-1:
                        outfile3.write('(%d,%d)\n' % (i,j))
                    else:
                        print("Error - no match : " + str(cube_labels[i,j]))
                        
    if dim ==3:
        with open(path_out + 'att0.txt','w') as outfile0, open(path_out + 'att1.txt','w') as outfile1, open(path_out + 'att2.txt','w') as outfile2, open(path_out + 'att3.txt','w') as outfile3, open(path_out + 'att4.txt','w') as outfile4, open(path_out + 'sep.txt','w') as outfile5:               
            for i in range(N-1):
                for j in range(N-1):
                    for k in range(N-1):
                        if cube_labels[i,j,k]==0:
                            outfile0.write('(%d,%d,%d)\n' % (i,j,k))
                        elif cube_labels[i,j,k]==1:
                            outfile1.write('(%d,%d,%d)\n' % (i,j,k))
                        elif cube_labels[i,j,k]==2:
                            outfile2.write('(%d,%d,%d)\n' % (i,j,k))
                        elif cube_labels[i,j,k]==3:
                            outfile3.write('(%d,%d,%d)\n' % (i,j,k))
                        elif cube_labels[i,j,k]==4:
                            outfile4.write('(%d,%d,%d)\n' % (i,j,k))
                        elif cube_labels[i,j,k]==-1:
                            outfile5.write('(%d,%d,%d)\n' % (i,j,k))
                        else:
                            print("Error - no match : " + str(cube_labels[i,j,k]))  
                            
    if dim ==4:
        with open(path_out + 'att0.txt','w') as outfile0, open(path_out + 'att1.txt','w') as outfile1, open(path_out + '/sep.txt','w') as outfile2:              
            for i0 in range(N-1):
                for i1 in range(N-1):
                    for i2 in range(N-1):
                        for i3 in range(N-1):
                            if cube_labels[i0,i1,i2,i3]==0:
                                outfile0.write('(%d,%d,%d,%d)\n' % (i0,i1,i2,i3))
                            elif cube_labels[i0,i1,i2,i3]==1:
                                outfile1.write('(%d,%d,%d,%d)\n' % (i0,i1,i2,i3))
                            elif cube_labels[i0,i1,i2,i3]==-1:
                                outfile2.write('(%d,%d,%d,%d)\n' % (i0,i1,i2,i3))
                            else:
                                print("Error - no match : " + str(cube_labels[i0,i1,i2,i3]))
                                
    if dim == 6:
        
        with open(path_out + 'att0.txt','w+') as outfile0, open(path_out + 'att1.txt','w') as outfile1, open(path_out + 'att2.txt','w') as outfile2, \
        open(path_out + 'att3.txt','w') as outfile3, open(path_out + 'att4.txt','w') as outfile4, open(path_out + 'att5.txt','w') as outfile5, \
        open(path_out + 'att6.txt','w') as outfile6, open(path_out + 'att7.txt','w') as outfile7, open(path_out + 'sep.txt','w') as outfile8:              
            for i0 in range(N-1):
                for i1 in range(N-1):
                    for i2 in range(N-1):
                        for i3 in range(N-1):
                            for i4 in range(N-1):
                                for i5 in range(N-1):
                                    if cube_labels[i0,i1,i2,i3,i4,i5]==0:
                                        outfile0.write('(%d,%d,%d,%d,%d,%d)\n' % (i0,i1,i2,i3,i4,i5))
                                    elif cube_labels[i0,i1,i2,i3,i4,i5]==1:
                                        outfile1.write('(%d,%d,%d,%d,%d,%d)\n' % (i0,i1,i2,i3,i4,i5))
                                    elif cube_labels[i0,i1,i2,i3,i4,i5]==2:
                                        outfile2.write('(%d,%d,%d,%d,%d,%d)\n' % (i0,i1,i2,i3,i4,i5))
                                    elif cube_labels[i0,i1,i2,i3,i4,i5]==3:
                                        outfile3.write('(%d,%d,%d,%d,%d,%d)\n' % (i0,i1,i2,i3,i4,i5))
                                    elif cube_labels[i0,i1,i2,i3,i4,i5]==4:
                                        outfile4.write('(%d,%d,%d,%d,%d,%d)\n' % (i0,i1,i2,i3,i4,i5))
                                    elif cube_labels[i0,i1,i2,i3,i4,i5]==5:
                                        outfile5.write('(%d,%d,%d,%d,%d,%d)\n' % (i0,i1,i2,i3,i4,i5))
                                    elif cube_labels[i0,i1,i2,i3,i4,i5]==6:
                                        outfile6.write('(%d,%d,%d,%d,%d,%d)\n' % (i0,i1,i2,i3,i4,i5))
                                    elif cube_labels[i0,i1,i2,i3,i4,i5]==7:
                                        outfile7.write('(%d,%d,%d,%d,%d,%d)\n' % (i0,i1,i2,i3,i4,i5))
                                    elif cube_labels[i0,i1,i2,i3,i4,i5]==-1:
                                        outfile8.write('(%d,%d,%d,%d,%d,%d)\n' % (i0,i1,i2,i3,i4,i5))
                                    else:
                                        print("Error - no match : " + str(cube_labels[i0,i1,i2,i3,i4,i5]))
            
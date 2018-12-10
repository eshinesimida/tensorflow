# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:11:23 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl

num = 200
l = np.linspace(-5,5,num)
X, Y = np.meshgrid(l, l) #meshgrid的作用适用于生成网格型数据，可以接受两个一维数组生成两个二维矩阵
#np.expand_dims增加一个维度(下面是增加第三维)
#plt.plot(X[1],X[2])
pos = np.concatenate((np.expand_dims(X,axis=2),np.expand_dims(Y,axis=2)),axis=2)

def plot_multi_normal(u,sigma):
    fig = plt.figure(figsize=(4,4))
    ax = Axes3D(fig)

    a = (pos-u).dot(np.linalg.inv(sigma))   #np.linalg.inv()矩阵求逆
    b = np.expand_dims(pos-u,axis=3)
    Z = np.zeros((num,num), dtype=np.float32)
    for i in range(num):
        Z[i] = [np.dot(a[i,j],b[i,j]) for j in range(num)]
    Z = np.exp(Z*(-0.5))/(2*np.pi*(np.linalg.det(sigma))**(0.5))   #np.linalg.det()矩阵求行列式
    ax.plot_surface(X, Y, Z, rstride=5, cstride=5, alpha=0.4, cmap=mpl.cm.bwr)
    cset = ax.contour(X,Y,Z, zdir='z',offset=0,cmap=cm.coolwarm,alpha=0.8)  #contour画等高线
    cset = ax.contour(X, Y, Z, zdir='x', offset=-5,cmap=mpl.cm.winter,alpha=0.8)
    cset = ax.contour(X, Y, Z, zdir='y', offset= 5,cmap= mpl.cm.winter,alpha=0.8)
    ax.set_zlim([0,0.3])   
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
u = np.array([0, 0])
sigma = np.array([[1, 0],[0, 1]])
plot_multi_normal(u,sigma)

u = np.array([0, 0])
sigma = np.array([[0.8, 0],[0, 0.8]])
plot_multi_normal(u,sigma)

u = np.array([0, 0])
sigma = np.array([[1.5, 0],[0, 1.5]])
plt.figure(figsize = (3,3))
plt.subplot(221)
plot_multi_normal(u,sigma)

u = np.array([0, 0])
sigma = np.array([[1, 0.4],[0.4, 1]])
plt.subplot(222)
plot_multi_normal(u,sigma)

u = np.array([0, 0])
sigma = np.array([[1, 0.8],[0.8, 1]])
plot_multi_normal(u,sigma)

u = np.array([0, 0])
sigma = np.array([[1, -0.5],[-0.5, 1]])
plot_multi_normal(u,sigma)

u = np.array([1, 0])
sigma = np.array([[1, 0],[0, 1]])
plot_multi_normal(u,sigma)




import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl

num = 200
l = np.linspace(-5,5,num)
X, Y =np.meshgrid(l, l)
pos = np.concatenate((np.expand_dims(X,axis=2),np.expand_dims(Y,axis=2)),axis=2)


def plot_two_gaussian(u1,sigma1,u2,sigma2):
    fig = plt.figure(figsize=(4,4))
    ax = Axes3D(fig)

    a1 = (pos-u1).dot(np.linalg.inv(sigma1))
    b1 = np.expand_dims(pos-u1,axis=3)
    Z1 = np.zeros((num,num), dtype=np.float32)

    a2 = (pos-u2).dot(np.linalg.inv(sigma2))
    b2 = np.expand_dims(pos-u2,axis=3)
    Z2 = np.zeros((num,num), dtype=np.float32)

    for i in range(num):
        Z1[i] = [np.dot(a1[i,j],b1[i,j]) for j in range(num)]
        Z2[i] = [np.dot(a2[i,j],b2[i,j]) for j in range(num)]
    Z1 = np.exp(Z1*(-0.5))/(2*np.pi*(np.linalg.det(sigma1))**0.5)
    Z2 = np.exp(Z2*(-0.5))/(2*np.pi*(np.linalg.det(sigma2))**0.5)

    Z = Z1 + Z2

    ax.plot_surface(X, Y, Z, rstride=5, cstride=5, alpha=0.4, cmap=mpl.cm.bwr)
    cset = ax.contour(X,Y,Z, zdir='z',offset=0,cmap=cm.coolwarm,alpha=0.8)  #contour画等高线
    cset = ax.contour(X, Y, Z, zdir='x', offset=-5,cmap=mpl.cm.winter,alpha=0.8)
    cset = ax.contour(X, Y, Z, zdir='y', offset= 5,cmap= mpl.cm.winter,alpha=0.8)
    ax.set_zlim([0,0.3])   
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
u1 = np.array([1.3, 1.3])
sigma1 = np.array([[1, 0],[0, 1]])
u2 = np.array([-1.3, -1.3])
sigma2 = np.array([[1, 0],[0, 1]])

plot_two_gaussian(u1,sigma1,u2,sigma2)

import matplotlib.pyplot as plt
import numpy as np

#create two data sets
m0 = [2,3]
cov = np.mat([[1,0],[0,2]])
x0 = np.random.multivariate_normal(m0, cov, 500).T
y0 = np.zeros(x0.shape[1])
plt.scatter(x0[0],x0[1])
plt.hist(x0[0])
plt.hist(x0[1])

m1 = [7,8]
cov = np.mat([[1,0],[0,2]])
x1 = np.random.multivariate_normal(m1, cov, 300).T
y1 = np.ones(x1.shape[1])

x = np.concatenate((x0, x1), axis = 1)
y = np.concatenate((y0, y1), axis = 0)
m = x.shape[1]

plt.scatter(x[0], x[1])

#
phi = (1.0/m)*len(y1)
u0 = np.mean(x0, axis = 1)
u1 = np.mean(x1, axis = 1)

xplot0 = x0
xplot1 = x1

x0 = x0.T
x1 = x1.T
x = x.T

x0_sub_u0 = x0 - u0
x1_sub_u1 = x1 - u1

x_sub_u = np.concatenate([x0_sub_u0,x1_sub_u1])
x_sub_u = np.mat(x_sub_u)

sigma = (1.0/m)*(x_sub_u.T*x_sub_u)

midPoint=[(u0[0]+u1[0])/2.0,(u0[1]+u1[1])/2.0]
k=(u1[1]-u0[1])/(u1[0]-u0[0])
x=range(-2,11)
y=[(-1.0/k)*(i-midPoint[0])+midPoint[1] for i in x]

#画高斯判别的contour
def gaussian_2d(x, y, x0, y0, sigmaMatrix):
    return np.exp(-0.5*((x-x0)**2+0.5*(y-y0)**2))
delta = 0.025
xgrid0=np.arange(-2, 6, delta)
ygrid0=np.arange(-2, 6, delta)
xgrid1=np.arange(3,11,delta)
ygrid1=np.arange(3,11,delta)
X0,Y0=np.meshgrid(xgrid0, ygrid0)   #generate the grid
X1,Y1=np.meshgrid(xgrid1,ygrid1)
Z0=gaussian_2d(X0,Y0,2,3,cov)
Z1=gaussian_2d(X1,Y1,7,8,cov)

plt.figure(figsize=(12,9))
plt.clf()
plt.plot(xplot0[0],xplot0[1],'ko')
plt.plot(xplot1[0],xplot1[1],'gs')
plt.plot(u0[0],u0[1],'rx',markersize=20)
plt.plot(u1[0],u1[1],'y*',markersize=20)
plt.plot(x,y)
CS0=plt.contour(X0, Y0, Z0)
plt.clabel(CS0, inline=1, fontsize=10)
CS1=plt.contour(X1,Y1,Z1)
plt.clabel(CS1, inline=1, fontsize=10)
plt.title("Gaussian discriminat analysis")
plt.xlabel('Feature Dimension (0)')
plt.ylabel('Feature Dimension (1)')
plt.show()


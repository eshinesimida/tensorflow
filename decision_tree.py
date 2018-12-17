# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:45:11 2018

@author: Administrator
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pydotplus
from sklearn import datasets
from sklearn.metrics import accuracy_score


iris_feature_E = 'sepal length', 'speal width', 'petal length','petal width'
iris_feature = u'花萼长度',u'花萼宽度',u'花瓣长度',u'花瓣宽度'
iris_class = 'Iris-setosa','Iris-versicolor','Iris-virginica'




if __name__ == 'main':
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    
    iris = datasets.load_iris()
    X = iris.data[:, [2,3]]
    y = iris.target
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 1)
    
    model = DecisionTreeClassifier(criterion = 'entropy')
    model.fit(x_train, y_train)
    y_test_hat = model.predict(x_test)
    print('acc:',accuracy_score(y_test, y_test_hat))
    
    #save
    with open('iris.dot', 'w') as f:
        tree.export_graphviz(model, out_file = f)
    
    dot_data = tree.export_graphviz(model, out_file = None, feature_names = iris.feature_names[2:4],
                                    class_names = iris_class, filled = True, rounded = True,
                                    special_characters = True)
    
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf('iris.pdf')
    f = open('iris.png', 'wb')
    f.write(graph.create_png())
    f.close()
    
    #picture
    N,M = 50,50
    x1_min,x2_min = np.min(X[:,0]),np.min(X[:,1])
    x1_max,x2_max = np.max(X[:,0]),np.max(X[:,1])
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)
    x_show = np.stack((x1.flat, x2.flat), axis = 1)
    print(x_show.shape)
    
    cm_light = mpl.colors.ListedColormap(['#A0FFA0','#A0A0FF','#FFA0A0'])
    cm_dark = mpl.colors.ListedColormap(['r','g','b'])
    y_show_hat = model.predict(x_show)
    print(y_show_hat.shape)
    print(y_show_hat)
    y_show_hat = y_show_hat.reshape(x1.shape)
    print(y_show_hat)
    plt.figure(facecolor = 'w')
    plt.pcolormesh(x1, x2, y_show_hat, cmap = cm_light)
    plt.scatter(x_test[:,0], x_test[:,1], c = y_test.ravel(), edgecolors = 'k',
                s = 150, zorder = 10, cmap = cm_dark, marker = '*')
    plt.scatter(X[:,0], X[:,1], c = y.ravel(), edgecolors = 'k', s = 40,
                cmap = cm_dark) 
    plt.xlabel(iris_feature[2],fontsize = 15)
    plt.ylabel(iris_feature[3], fontsize = 15)
    plt.xlim(x1_min,x1_max)
    plt.ylim(x2_min,x2_max)
    plt.grid(True)
    plt.show()
    
    #traininig data predict results
    y_test = y_test.reshape(-1)
    print(y_test_hat)
    print(y_test)
    result = (y_test_hat == y_test)
    acc = np.mean(result)
    print('acc: %.2f%%' %(100 * acc) )
    
    #overfit , error rate
    depth = np.arange(1, 15)
    err_list = []
    for d in depth:
        clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = d)
        clf.fit(x_train, y_train)
        y_test_hat = clf.predict(x_test)
        result = (y_test_hat == y_test)
        if d == 1:
            print(result)
        err = 1 - np.mean(result)
        err_list.append(err)
        print(d, 'err rate %.2f%%' % (100 * err))
    
    plt.figure(facecolor = 'w')
    plt.plot(depth, err_list, 'ro-', lw = 2)
    plt.xlabel('num_depth', fontsize = 15)
    plt.ylabel('err_rate', fontsize = 15)
    plt.title('depth and overfit', fontsize = 15)
    plt.grid(True)
    plt.show()
        
    
    
                                  
                                          
                                         
    
    
    #picture entropy and gini and error 
    
    p = np.arange(0.001, 1, 0.001, dtype = np.float)
    gini = 2 * p * (1-p)
    h = -(p * np.log2(p) + (1-p)*np.log2(1-p))/2
    err = 1 - np.max(np.vstack((p, 1-p)), 0)
    plt.plot(p, h, 'b-', linewidth = 2, label = 'Entropy')
    plt.plot(p, gini, 'r-', linewidth = 2, label = 'Gini')
    plt.plot(p, err, 'g-', linewidth = 2, label = 'Err')
    plt.grid(True)
    plt.legend(loc = 'upper left')
    plt.show()
    
 

            
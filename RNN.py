# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 10:10:47 2018

@author: Administrator
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#mnist=input_data.read_data_sets("D:\BaiDu\MNIST_data",one_hot=True)
mnist = input_data.read_data_sets('mnist_data', one_hot = True)
 
#每个批次的大小
batch_size=100
#计算一共有多少个批次
n_batch=mnist.train.num_examples//batch_size
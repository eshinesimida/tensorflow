# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 13:19:42 2018

@author: Administrator
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#load data
mnist = input_data.read_data_sets('mnist_data', one_hot = True)

# size of batch
batch_size  = 100
#all size
n_batch = mnist.train.num_examples // batch_size

#define two placeholder
x = tf.placeholder(tf.float32,[None, 784])
y = tf.placeholder(tf.float32,[None, 10])

#define nn
W = tf.Variable(tf.ones([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W) + b)

#L2
#W1 = tf.Variable(tf.random_normal([100,10]))
#b1 = tf.Variable(tf.zeros([10]))
#prediction_value = tf.nn.softmax(tf.matmul(prediction, W1) + b1)

#loss funciton
loss = tf.reduce_mean(tf.square(y - prediction))

#GD
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()


# result in bool list
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction, 1))
#acc
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batchys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict = { x:batch_xs, y:batchys})
        
        acc = sess.run(accuracy, feed_dict = {x:mnist.test.images,
                                              y:mnist.test.labels})
        print('lter:' + str(epoch) + ', Testing Accuracy:' + str(acc))
        #print(sess.run(b))
        
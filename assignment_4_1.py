# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 08:57:54 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:49:55 2018

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
lr = tf.Variable(0.001, dtype = tf.float32)
keep_prob = tf.placeholder(tf.float32)

#define nn
W1 = tf.Variable(tf.truncated_normal([784,500], stddev = 0.1))
b1 = tf.Variable(tf.zeros([500]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x,W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([500,300], stddev = 0.1))
b2 = tf.Variable(tf.zeros([300]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([300,100], stddev = 0.1))
b3 = tf.Variable(tf.zeros([100]) + 0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop,W3) + b3)
L3_drop = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.truncated_normal([100,10], stddev = 0.1))
b4 = tf.Variable(tf.zeros([10]) + 0.1)

prediction = tf.nn.softmax(tf.matmul(L3_drop,W4) + b4)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction))

#GD
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
#train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()


# result in bool list
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction, 1))
#acc
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(51):
        #tf.assign(lr, 0.01)
        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
        for batch in range(n_batch):
            batch_xs,batchys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict = { x:batch_xs, y:batchys,
                                              keep_prob:1.0})
        
        test_acc = sess.run(accuracy, feed_dict = {x:mnist.test.images,
                                              y:mnist.test.labels,
                                              keep_prob:1.0})
        train_acc = sess.run(accuracy, feed_dict = {x:mnist.train.images,
                                              y:mnist.train.labels,
                                              keep_prob:1.0})
        print('lter:' + str(epoch) + ', Testing Accuracy:' + str(test_acc) +
              ', Train Accuracy:' + str(train_acc))
        #print(sess.run(b))
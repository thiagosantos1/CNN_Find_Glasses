"""
Author: Thiago Santos
Master thesis project - Fall 2018

We will use convolution layers, each one followed by a max pooling. At the end, we flatten the results, 
and give to a fully connected layer with one hidden layer. We will use a simplifcation of a restnet architecture.
"""

#!/usr/bin/env python3
# chmod u+x
# -*- coding: utf-8 -*-  


import tensorflow as tf
from idx3_format import load_img_lbl_idx3
from idx3_format import display_img
from sklearn.model_selection import train_test_split
import numpy as np
import os
import random
import pandas as pad


# create a weight variable - Filter
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1) # to init with a random normal distribution with stndard deviation of 1
  return tf.Variable(initial) # create a variable

# create a bias variable
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# execute and return a convolutional over a data x, with a filter/weights W
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# create, execute and return the result of a convolutional layer, already with an activation 
def conv_layer(input_, shape):
  W = weight_variable(shape)
  b = bias_variable([shape[3]])
  return tf.nn.relu(conv2d(input_, W) + b)

# with X as a result of a convolutional layer, we will max pool
# with a filter of 2x2
# this basically gets the most important features, and reduces the size of the inputs for the final densed layer
def max_pool_2x2(x, ksize_=[1, 2, 2, 1], strides_=[1, 2, 2, 1]):
  return tf.nn.max_pool(x, ksize=ksize_, strides=strides_, padding='SAME')

# After all convolutional layers has been applied, we get all the final results, and make a full connected layer
def full_layer(input, size):
  in_size = int(input.get_shape()[1])
  W = weight_variable([in_size, size])
  b = bias_variable([size])
  # tf.matmul is a matrix multiplication from tensorn. This is the basic idea of ML
  # multiply 2 matrix and add a bias. This is the foward when we implement ANN
  return tf.matmul(input, W) + b


# the idea is to select a random part of the data, with size = MINIBATCH_SIZE
def next_batch(MINIBATCH_SIZE, X_train, y_train, size_train):
  l_bound = random.randint(0,size_train - MINIBATCH_SIZE)
  u_bound = l_bound + MINIBATCH_SIZE

  return X_train[l_bound:u_bound], y_train[l_bound:u_bound]

def main(): 

  #### Data pre-processing
  X_all, y_all = load_img_lbl_idx3(dataset='all', path='dataset', rotate=True) 
  X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, shuffle=True)

  size_train = len(X_train)
  size_test = len(X_test)
  size_classes = len(y_train[0])
  height_pic = len(X_train[0])
  width_pics = len(X_train[0][0])
  # flatten each picture of Training and Testing images
  X_train = X_train.reshape(size_train, height_pic*width_pics)
  X_test = X_test.reshape(size_test, height_pic*width_pics)

  # HYPER paramters
  NUM_STEPS = 40000
  MINIBATCH_SIZE = 40
  learning_rate_ = 0.00001
  size_hidden_layer = 64

  #### create model architecture ####

  # placeholder for the data training. One neuron for each pixel
  x = tf.placeholder(tf.float32, [None, height_pic*width_pics])
  # correct output
  y = tf.placeholder(tf.float32, [None, size_classes])

  # reshape for WxH, and only macro - 1 channel
  x_image = tf.reshape(x, [-1, height_pic, width_pics, 1])

  # create first convolutional followed by pooling
  conv1 = conv_layer(x_image, shape=[5, 5, 1, 32]) # in this case, a filter of 5x5, used 32 times over the image
  # the result of conv1, which is 112x112x32, we feed to pooling
  conv1_pool = max_pool_2x2(conv1) # the result of this first polling will be 56X56X32

  # create second convolutional followed by pooling. 32 came from the first convol
  conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64]) # the result here will be 56X56X64
  conv2_pool = max_pool_2x2(conv2) # the result will be 28X28X64

  # create a third layer
  conv3 = conv_layer(conv2_pool, shape=[5, 5, 64, 128]) # the result here will be 28X28X128
  conv3_pool = max_pool_2x2(conv3) # the result will be 14X14X128

  # create a forth layer
  conv4 = conv_layer(conv3_pool, shape=[5, 5, 128, 256]) # the result here will be 14X14X256
  conv4_pool = max_pool_2x2(conv4) # the result will be 7X7X256

  # flat the final results, for then put in a fully connected layer
  # since the result data is 28X23X64 and we want to flat, Just a big array
  conv5_flat = tf.reshape(conv4_pool, [-1, 7*7*256])

  # create fully connected layer and train - Foward
  full_1 = tf.nn.relu(full_layer(conv5_flat, size_hidden_layer)) 

  # for dropout
  keep_prob = tf.placeholder(tf.float32)
  full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob) # for test, we will use full drop(no drops)

  # for output - For training
  # In this case, weights will have size of 10 - Because we have 10 classes as output
  y_conv = full_layer(full1_drop, size_classes) # one last layer, for the outputs

  # error function. Using cross entropy to calculate the distance between probabilities
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y))

  # define which optimezer to use. How to change bias and weights to get to the result
  optimizer_train = tf.train.AdamOptimizer(learning_rate=learning_rate_).minimize(cross_entropy)

  # correct prediction
  correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))

  # check for accuracy
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  # Training Loop
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(NUM_STEPS):
      batch_xs, batch_ys = next_batch(MINIBATCH_SIZE,X_train, y_train, size_train)
      if i % 100 == 0: # to print the results every 100 steps
        train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        print("step {}, training accuracy {}".format(i, train_accuracy))

      sess.run(optimizer_train, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.75})

    batch_xs, batch_ys = next_batch(80,X_test, y_test, size_test)
    test_accuracy = sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
    print("test accuracy: {}".format(test_accuracy))


if __name__ == '__main__':
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
  tf.logging.set_verbosity(tf.logging.ERROR)
  main()





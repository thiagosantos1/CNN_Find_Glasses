#!/usr/bin/env python3
# chmod u+x
# -*- coding: utf-8 -*- 

"""
Author: Thiago Santos
Master thesis project - Fall 2018

The idea is to test our model with only 1 layer(full layer of perceptrons), without hidden or convolution layers. 
Our goal is to get a better result than if a computer were just guessing.
"""

import tensorflow as tf
from idx3_format import load_img_lbl_idx3
import numpy as np
import os
import random

# the idea is to select a random part of the data, with size = MINIBATCH_SIZE
def next_batch(MINIBATCH_SIZE, X_train, y_train, size_train):
  l_bound = random.randint(0,size_train - MINIBATCH_SIZE)
  u_bound = l_bound + MINIBATCH_SIZE

  return X_train[l_bound:u_bound], y_train[l_bound:u_bound]

if __name__ == '__main__':

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
  tf.logging.set_verbosity(tf.logging.ERROR)

  #### Data pre-processing
  X_train, y_train = load_img_lbl_idx3(dataset='training', path='dataset')
  X_test, y_test = load_img_lbl_idx3(dataset='testing', path='dataset')

  size_train = len(X_train)
  size_test = len(X_test)
  size_classes = len(y_train[0])
  height_pic = len(X_train[0])
  width_pics = len(X_train[0][0])

  # flatten each picture of Training and Testing images
  X_train = X_train.reshape(size_train, height_pic*width_pics)
  X_test = X_test.reshape(size_test, height_pic*width_pics)

  # Training paramters
  NUM_STEPS = 10000
  MINIBATCH_SIZE = 20
  learning_rate_ = 0.0001

  # placeholder for the data training. One neuron for each pixel
  x = tf.placeholder(tf.float32, [None, height_pic*width_pics])

  # Variable for the baias. One for each neuron - This is what we will calculate/change
  W = tf.Variable(tf.zeros([height_pic*width_pics, size_classes]))

  # correct output
  y = tf.placeholder(tf.float32, [None, size_classes])
  
  # predicted output
  y_pred = tf.matmul(x, W)

  # error function. Using cross entropy to calculate the distance between probabilities
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))

  # define which optimezer to use. How to change bias and weights to get to the result
  optimizer_train = tf.train.AdamOptimizer(learning_rate=learning_rate_).minimize(cross_entropy)

  # correct prediction
  correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))

  # check for accuracy
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  # Training Loop
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(NUM_STEPS):
      batch_xs, batch_ys = next_batch(MINIBATCH_SIZE,X_train, y_train, size_train)
      sess.run(optimizer_train, feed_dict={x: batch_xs, y: batch_ys})

    ans = sess.run(accuracy, feed_dict={x: X_test, y: y_test})
    print(ans)






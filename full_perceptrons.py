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
from sklearn.model_selection import train_test_split
from convolution_manual import *


# the idea is to select a random part of the data, with size = MINIBATCH_SIZE
def next_batch(MINIBATCH_SIZE, X_train, y_train, size_train):
  l_bound = random.randint(0,size_train - MINIBATCH_SIZE)
  u_bound = l_bound + MINIBATCH_SIZE

  return X_train[l_bound:u_bound], y_train[l_bound:u_bound]

def display_W(W,sess,size,perc=0.05):
  t = sess.run(W[:,0])
  k = int(size * perc)
  idx = np.argpartition(t, k-1)
  out = np.zeros(size, dtype=int)
  for x in range(k):
    out[idx[x]] = 1

  out = out.reshape(112,112)
  display_img(out)

  t = sess.run(W[:,1])
  k = int(size * perc)
  idx = np.argpartition(t, k-1)
  out = np.zeros(size, dtype=int)
  for x in range(k):
    out[idx[x]] = 1

  out = out.reshape(112,112)
  display_img(out)


if __name__ == '__main__':

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
  tf.logging.set_verbosity(tf.logging.ERROR)
  save_to="../saved_percept/model.ckpt"
  number_files = len(os.listdir("../saved_percept"))

  sess = tf.Session()
  disp_W = True

  #### Data pre-processing
  X_all, y_all = load_img_lbl_idx3(dataset='all', path='/u1/h2/tsantos2/695_Projects/CNN_Find_Glasses/dataset', rotate=False) 
  X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, shuffle=True)

  size_train = len(X_train)
  size_test = len(X_test)
  size_classes = len(y_train[0])
  height_pic = len(X_train[0])
  width_pics = len(X_train[0][0])

  # for x in range(0,size_train,100):
  #   display_img(X_train[x])
  #   print(y_train[x])

  # flatten each picture of Training and Testing images
  X_train = X_train.reshape(size_train, height_pic*width_pics)
  X_test = X_test.reshape(size_test, height_pic*width_pics)

  # Training paramters
  NUM_STEPS = 40000
  MINIBATCH_SIZE = 25
  learning_rate_ = 0.00001

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

  #Training Loop
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  if number_files >0:
    saver.restore(sess, save_to)

  if len(sys.argv) < 2 and not disp_W:
    for i in range(NUM_STEPS):
      batch_xs, batch_ys = next_batch(MINIBATCH_SIZE,X_train, y_train, size_train)
      sess.run(optimizer_train, feed_dict={x: batch_xs, y: batch_ys})

      if i % 4000 == 0: # to print the results every 100 steps
        train_accuracy = sess.run(accuracy, feed_dict={x: X_test, y: y_test})
        saver.save(sess, save_to)
        print("step {}, training accuracy {}".format(i, train_accuracy))

    # Test accuracy
    saver.save(sess, save_to)
    ans = sess.run(accuracy, feed_dict={x: X_test, y: y_test})
    print(ans)

  elif not disp_W:  
    img = read_image(sys.argv[1])
    width,height = get_width_height(img)
    #output_img_ReLu = convolutional(img,width,height,brightness=[])
    #display_img(img)
    pred = sess.run(y_pred, feed_dict={x: [img.reshape(width*height)]})
    pred = tf.argmax(pred, 1)
    pred = sess.run(pred)[0]
    print(pred)

  if disp_W:
    size = height_pic*width_pics
    display_W(W,sess,size) # display the highest 20% values of W





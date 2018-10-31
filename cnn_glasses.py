#!/usr/bin/env python3
# chmod u+x
# -*- coding: utf-8 -*-  

"""
Author: Thiago Santos
Master thesis project - Fall 2018

We will use convolution layers, each one followed by a max pooling. At the end, we flatten the results, 
and give to a fully connected layer with one hidden layer. We will use a simplifcation of a restnet architecture.
"""


import tensorflow as tf
from idx3_format import load_img_lbl_idx3
from idx3_format import display_img
from convolution_manual import *
from sklearn.model_selection import train_test_split
import numpy as np
import os
import random
import pandas as pad
import os
import sys


class Model:

  sess = tf.Session()

  def __init__(self, X_train_, X_test_, y_train_, y_test_, NUM_STEPS_ = 40000, MINIBATCH_SIZE_ = 40, learning_rate_ = 0.00001, size_hidden_layer_ = 64):
    self.size_train = len(X_train_)
    self.size_test = len(X_test_)
    self.size_classes = len(y_train_[0])
    self.height_pic = len(X_train_[0])
    self.width_pics = len(X_train_[0][0])

    self.NUM_STEPS = NUM_STEPS_
    self.MINIBATCH_SIZE = MINIBATCH_SIZE_
    self.learning_rate = learning_rate_
    self.size_hidden_layer = size_hidden_layer_

    self.X_train = X_train_.reshape(self.size_train, self.height_pic * self.width_pics)
    self.X_test  = X_test_.reshape(self.size_test, self.height_pic * self.width_pics)
    self.y_train = y_train_
    self.y_test  = y_test_

    # set to true if it's the first training - Not reseting a graph
    self.first_training = True


    #### create model architecture ####

    # if had created some graph before/training, load it
    number_files = len(os.listdir("saved_model"))
    if (number_files > 0 ):
      #tf.reset_default_graph()
      self.first_training = False # reseting a graph

    # placeholder for the data training. One neuron for each pixel
    self.x = tf.placeholder(tf.float32, [None, self.height_pic * self.width_pics])
    # correct output
    self.y = tf.placeholder(tf.float32, [None, self.size_classes])

    # reshape for WxH, and only macro - 1 channel
    self.x_image = tf.reshape(self.x, [-1, self.height_pic, self.width_pics, 1])

    # create first convolutional followed by pooling
    self.conv1 = self.conv_layer(self.x_image, shape=[5, 5, 1, 32]) # in this case, a filter of 5x5, used 32 times over the image
    # the result of conv1, which is 112x112x32, we feed to pooling
    self.conv1_pool = self.max_pool_2x2(self.conv1) # the result of this first polling will be 56X56X32

    # create second convolutional followed by pooling. 32 came from the first convol
    self.conv2 = self.conv_layer(self.conv1_pool, shape=[5, 5, 32, 64]) # the result here will be 56X56X64
    self.conv2_pool = self.max_pool_2x2(self.conv2) # the result will be 28X28X64

    # create a third layer
    self.conv3 = self.conv_layer(self.conv2_pool, shape=[5, 5, 64, 128]) # the result here will be 28X28X128
    self.conv3_pool = self.max_pool_2x2(self.conv3) # the result will be 14X14X128

    # create a forth layer
    self.conv4 = self.conv_layer(self.conv3_pool, shape=[5, 5, 128, 256]) # the result here will be 14X14X256
    self.conv4_pool = self.max_pool_2x2(self.conv4) # the result will be 7X7X256

    # flat the final results, for then put in a fully connected layer
    # since the result data is 28X23X64 and we want to flat, Just a big array
    self.conv5_flat = tf.reshape(self.conv4_pool, [-1, 7*7*256])

    # create fully connected layer and train - Foward
    self.full_1 = tf.nn.relu(self.full_layer(self.conv5_flat, self.size_hidden_layer))

    # for dropout
    self.keep_prob = tf.placeholder(tf.float32)
    self.full1_drop = tf.nn.dropout(self.full_1, keep_prob=self.keep_prob) # for test, we will use full drop(no drops)

    # for output - For training
    # In this case, weights will have size of 10 - Because we have 10 classes as output
    self.y_conv = self.full_layer(self.full1_drop, self.size_classes) # one last layer, for the outputs

    # error function. Using cross entropy to calculate the distance between probabilities
    self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_conv, labels=self.y))

    # define which optimezer to use. How to change bias and weights to get to the result
    self.optimizer_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy)

    # correct prediction
    self.correct_pred = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y, 1))

    # check for accuracy
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32)) 


  # create a weight variable - Filter
  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # to init with a random normal distribution with stndard deviation of 1
    return tf.Variable(initial) # create a variable

  # create a bias variable
  def bias_variable(self, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  # execute and return a convolutional over a data x, with a filter/weights W
  def conv2d(self, x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  # create, execute and return the result of a convolutional layer, already with an activation 
  def conv_layer(self, input_, shape):
    W = self.weight_variable(shape)
    b = self.bias_variable([shape[3]])
    return tf.nn.relu(self.conv2d(input_, W) + b)

  # with X as a result of a convolutional layer, we will max pool
  # with a filter of 2x2
  # this basically gets the most important features, and reduces the size of the inputs for the final densed layer
  def max_pool_2x2(self, x, ksize_=[1, 2, 2, 1], strides_=[1, 2, 2, 1]):
    return tf.nn.max_pool(x, ksize=ksize_, strides=strides_, padding='SAME')

  # After all convolutional layers has been applied, we get all the final results, and make a full connected layer
  def full_layer(self, input, size):
    in_size = int(input.get_shape()[1])
    W = self.weight_variable([in_size, size])
    b = self.bias_variable([size])
    # tf.matmul is a matrix multiplication from tensorn. This is the basic idea of ML
    # multiply 2 matrix and add a bias. This is the foward when we implement ANN
    return tf.matmul(input, W) + b

  # the idea is to select a random part of the data, with size = MINIBATCH_SIZE
  def next_batch(self, data, label_data, size_data):
    l_bound = random.randint(0,size_data - self.MINIBATCH_SIZE)
    u_bound = l_bound + self.MINIBATCH_SIZE

    return data[l_bound:u_bound], label_data[l_bound:u_bound]

  def restore_model(self, load_from="saved_model/model.ckpt"):

    # create a variable to load trained graph
    saver = tf.train.Saver()

    # start from where we left last time
    if (self.first_training == True):
      print("You must train the model first")
      sys.exit()

    # restore Graph trained
    saver.restore(self.sess, load_from)

  def training(self, save_to="saved_model/model.ckpt"):
    
    # initialize variable
    self.sess.run(tf.global_variables_initializer())

    # create a variable to save graph
    saver = tf.train.Saver()

    # start from where we left last time
    if (self.first_training == False):
      saver.restore(self.sess, save_to)

    for i in range(self.NUM_STEPS):
      batch_xs, batch_ys = self.next_batch(self.X_train, self.y_train, self.size_train)
      if i % 100 == 0: # to print the results every 100 steps
        train_accuracy = self.sess.run(self.accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys, self.keep_prob: 1.0})
        print("step {}, training accuracy {}".format(i, train_accuracy))

      self.sess.run(self.optimizer_train, feed_dict={self.x: batch_xs, self.y: batch_ys, self.keep_prob: 0.75})

    #save new training
    saver.save(self.sess, save_to)

  def test(self, img=[], label=[], num_steps=1, load_from="saved_model/model.ckpt"):

    # restore model with last training
    self.restore_model()

    batch_xs, batch_ys = img, label
    test_accuracy = 0
    for i in range(num_steps):
      if len(img) <=0 and len(label) <=0:
        batch_xs, batch_ys = self.next_batch(self.X_test, self.y_test, self.size_test)

      test_accuracy += self.sess.run(self.accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys, self.keep_prob: 1.0})

    print("test accuracy: {}".format(test_accuracy/num_steps))

  # test just for one image
  def single_test(self, img):

    # restore model with last training
    self.restore_model()

    pred = self.sess.run(self.y_conv, feed_dict={self.x: [img.reshape(width*height)], self.keep_prob: 1.0})
    pred = tf.argmax(pred, 1)
    
    # 0 for non-glasses and 1 for glasses
    return self.sess.run(pred)[0]


if __name__ == '__main__':

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
  tf.logging.set_verbosity(tf.logging.ERROR)

  #### Data pre-processing
  X_all, y_all = load_img_lbl_idx3(dataset='all', path='dataset', rotate=True) 
  X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, shuffle=True)

  model = Model(X_train, X_test, y_train, y_test, NUM_STEPS_ = 1, MINIBATCH_SIZE_ = 5, learning_rate_ = 0.001, size_hidden_layer_ = 5 )
  model.training()
  model.test( [model.X_test[0]], [model.y_test[0]])
  #model.test(num_steps=5)

  #img = read_image("dataset/faces_original/1/faces_2638.pgm")
  img = read_image("bill_2.pgm")
  width,height = get_width_height(img)
  output_img_ReLu = convolutional(img,width,height)
  
  prediction = model.single_test(output_img_ReLu)

  print(prediction)

  """
    # trying to get some pics of what the network is looking at
    a = sess.run(conv1_pool, feed_dict={x: [X_test[0]], y: [y_test[0]], keep_prob: 1.0})
    print( a.shape)
    c = np.transpose(a,[0,3,1,2])
    print( c.shape)
    d = c[0,0,:,:]
    display_img(X_test[0].reshape(112,112))
    display_img(d)
  """


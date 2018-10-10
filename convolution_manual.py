#!/usr/bin/env python3
# chmod u+x
# -*- coding: utf-8 -*-  

import matplotlib.pyplot as plt
#from scipy import misc
import numpy as np
from math import ceil
import random
import imageio 
import sys
from math import ceil
import scipy.misc
import os
from idx3_format import convert_training_test_to_idx3
from idx3_format import convert_all_imgs_to_idx3
from idx3_format import load_img_lbl_idx3

def read_images(img_path):
  try:
    img = np.array(imageio.imread(img_path), dtype=np.uint8)
  except:
    print("Img " + str(img_path) + " do not exist")
    sys.exit(1)

  return img

def display_img(img):
  # Plot the input
  plt.subplot(223)
  plt.imshow(img, cmap=plt.cm.gray)
  plt.axis('off')
  plt.show()

def display_all_idx3_img():
  imgs,lbs = load_img_lbl_idx3(path='dataset')
  for i in range(len(imgs)):
    display_img(imgs[i])

def save_img(img, path_to='dataset/out.pgm'):
  imageio.imwrite(path_to, img[:, :])

def get_width_height(img):
  return [len(img[0]), len(img)]


def convolutional(img, width,height, filter_conv=[[-1,-1,-1], [-1,4,-1], [-1,-1,-1]], h_stride=1, v_tride=1, paddings=0, out_half_size=False):

  fw = len(filter_conv[0])
  fh = len(filter_conv)

  w_out = ceil( ( (width - fw + (2*paddings)) // h_stride) + 1)
  h_out = ceil( ( (height - fh + (2*paddings)) // v_tride) + 1)

  output_img_conv = np.zeros((h_out,w_out),dtype=np.uint8)

  index_h = index_w = 0
  sum_dot = 0
  # for all "smalls" filters going trough the image
  for line_height in range(0,h_out,h_stride):
    for line_weidth in range(0,w_out,v_tride):
      sum_dot = 0
      # inside of the small part part image
      # use these indexes for the filter
      for pixel_height in range(fh):
        for pixel_weight in range(fw):
          sum_dot += filter_conv[pixel_height][pixel_weight] * img[line_height+pixel_height][line_weidth+pixel_weight]

      # ReLu the pixel, and make sure it goes just till 255, cause we are working with chromatic pics
      sum_dot = max(0,min(sum_dot,255))
      output_img_conv[line_height][line_weidth] = sum_dot

  # return half of the size
  if out_half_size:
    return  scipy.misc.imresize(np.pad(output_img_conv, (1,1), 'constant', constant_values=(0, 0)), (height//2, width//2))
  
  return np.pad(output_img_conv, (1,1), 'constant', constant_values=(0, 0))

def convert_all_convolution():

  #filter_conv=[[-1,-1,-1], [-1,8,-1], [-1,-1,-1]] # for edge and few more details
  filter_conv=[[0,-1,0], [-1,4,-1], [0,-1,0]] # for edge
  #filter_conv=[[1/16,1/8,1/16], [1/8,1/4,1/8], [1/16,1/8,1/16]] # - The information diffuses nearly equally among all pixels; 
                                                                # Gaussian blur or as Gaussian smoothing

  main_folder = ['dataset/faces_training_original','dataset/faces_test_original']
  folder_to   = ['dataset/faces_training','dataset/faces_test']

  try:
    index = 0
    for name in main_folder:
      for dirname in os.listdir(name): 
        path = os.path.join(name,dirname)
        for filename in os.listdir(path):
          if filename.endswith(".pgm"):
            img_path     = os.path.join(name,dirname,filename)
            save_path_to = os.path.join(folder_to[index],dirname,filename)
            img = read_images(img_path)
            width,height = get_width_height(img)
            output_img_ReLu = convolutional(img,width,height,filter_conv)
            save_img(output_img_ReLu, path_to=save_path_to)

      index += 1
  except:
    print('Folder do not exist');
    sys.exit(1)


if __name__ == '__main__':
  convert_all_convolution()
  convert_training_test_to_idx3()
  convert_all_imgs_to_idx3()









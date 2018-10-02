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

def read_images(img_path):
  
  try:
    img = np.array(imageio.imread(img_path), dtype=np.float64)
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

def save_img(img, path_to='../data_set_Faces/Faces_out/', out_name='out'):
  out=path_to+out_name+".pgm"
  imageio.imwrite(out, img[:, :])

def get_width_height(img):
  return [len(img[0]), len(img)]


def convolutional(img, width,height, filter_conv=[[-1,-1,-1], [-1,4,-1], [-1,-1,-1]], h_stride=1, v_tride=1, paddings=0):

  fw = len(filter_conv[0])
  fh = len(filter_conv)

  w_out = ceil( ( (width - fw + (2*paddings)) // h_stride) + 1)
  h_out = ceil( ( (height - fh + (2*paddings)) // v_tride) + 1)

  output_img = np.zeros((h_out,w_out))

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
      output_img[line_height][line_weidth] = sum_dot


  return output_img

def main():

  #filter_conv=[[-1,-1,-1], [-1,8,-1], [-1,-1,-1]] # for edge and few more details
  filter_conv=[[0,-1,0], [-1,4,-1], [0,-1,0]] # for edge
  #filter_conv=[[1/16,1/8,1/16], [1/8,1/4,1/8], [1/16,1/8,1/16]] # - The information diffuses nearly equally among all pixels; 
                                                                # Gaussian blur or as Gaussian smoothing


  img_path = '../data_set_Faces/Faces/s1/1.pgm'
  out_name_ = 'out'

  if len(sys.argv) >= 3:
    img_path = sys.argv[1] 
    out_name_ = sys.argv[2] 

  elif len(sys.argv) >= 2:
    img_path = sys.argv[1] 

  img = read_images(img_path)

  #display_img(img)

  width,height = get_width_height(img)

  output_img_ReLu = convolutional(img,width,height,filter_conv)

  #display_img(output_img_ReLu)

  save_img(output_img_ReLu,out_name=out_name_)

if __name__ == '__main__':
  main()



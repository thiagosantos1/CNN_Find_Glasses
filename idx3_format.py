import os
from PIL import Image
from array import *
from random import shuffle
import numpy as np
import imageio
import matplotlib.pyplot as plt
import struct
from struct import *


def convert_all():
  # Load from and save to
  Names = [['dataset/faces_training','train'], ['dataset/faces_test','test']]

  for name in Names:
    
    data_image = array('B')
    data_label = array('B')

    FileList = []
    for dirname in os.listdir(name[0]):     #[1:]: # [1:] Excludes .DS_Store from Mac OS
      path = os.path.join(name[0],dirname)
      for filename in os.listdir(path):
        if filename.endswith(".pgm"):
          FileList.append(os.path.join(name[0],dirname,filename))

    shuffle(FileList) # Usefull for further segmenting the validation set
    width = 0
    height = 0
    for filename in FileList:

      label = int(filename.split('/')[2])

      img = np.array(imageio.imread(filename), dtype=np.int64)

      height = len(img)
      width = len(img[0])
      for x in range(0,(width)):
        for y in range(0,(height)):
          data_image.append(img[y,x])

      data_label.append(label) # labels start (one unsigned byte each)
    hexval = "{0:#0{1}x}".format(len(FileList),6) # number of files in HEX

    # header for label array

    header = array('B')
    header.extend([0,0,8,1,0,0])
    header.append(int('0x'+hexval[2:][:2],16))
    header.append(int('0x'+hexval[2:][2:],16))
    
    data_label = header + data_label

    # additional header for images array
    
    if max([width,height]) <= 256:
      header.extend([0,0,0,width,0,0,0,height])
    else:
      raise ValueError('Image exceeds maximum size: 256x256 pixels');

    header[3] = 3 # Changing MSB for image data (0x00000803)
    
    data_image = header + data_image

    output_file = open('dataset/' + name[1] + '-images-idx3-ubyte', 'wb')
    data_image.tofile(output_file)
    output_file.close()

    output_file = open('dataset/' + name[1] + '-labels-idx1-ubyte', 'wb')
    data_label.tofile(output_file)
    output_file.close()

def display_img(img):
  # Plot the input
  plt.subplot(223)
  plt.imshow(img, cmap=plt.cm.gray)
  plt.axis('off')
  plt.show()

def load_img_lbl_idx3(dataset="training", classes=np.arange(2), path=".", size = 400):
  if dataset == "training":
      fname_img = os.path.join(path, 'train-images-idx3-ubyte')
      fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
  elif dataset == "testing":
      fname_img = os.path.join(path, 'test-images-idx3-ubyte')
      fname_lbl = os.path.join(path, 'test-labels-idx1-ubyte')

  else:
      raise ValueError("dataset must be 'testing' or 'training'")

  flbl = open(fname_lbl, 'rb')
  magic_nr, size = struct.unpack(">II", flbl.read(8))
  lbl = array("b", flbl.read())
  flbl.close()

  fimg = open(fname_img, 'rb')
  magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
  img = array("B", fimg.read())

  fimg.close()

  ind = [ k for k in range(size) if lbl[k] in classes ]
  N = size 
  images = np.zeros((N, rows, cols), dtype=np.uint8)
  images_out = np.zeros((N, cols, rows), dtype=np.uint8)
  labels = np.zeros((N, 1), dtype=np.int8)
  for i in range(N): #int(len(ind) * size/100.)):
    images[i] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])\
                .reshape((rows, cols))
    images_out[i] = np.rot90(images[i],3)
    labels[i] = lbl[ind[i]]
  labels = [label[0] for label in labels]

  return images_out, labels

if __name__ == '__main__':
  convert_all()
  imgs,lbs = load_img_lbl_idx3(path='dataset')
  for i in range(len(imgs)):
    pass#print(lbs[i])
    #display_img(imgs[i])




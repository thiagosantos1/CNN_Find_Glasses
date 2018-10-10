# CNN Find Glasses in Faces

# Main Idea

The idea of this project is to develop a CNN to recognize glasses in Faces.

# Steps

1) Run a manual convolutional with a filter for edge detecter and save the new pictures

2) Convert all convoluted images to idx3 - Ubyte 

3) Feed the final result to our models( perceptrons, CNN, ANN and others) to classify a picture as having glass or not.


# Code Data-Preprocess

* In this project we have 2 different python codes for Data-Preprocess:

  1) idx3_format.py --> Used to convert a given list of images to idx3 format(MNIST format). We also have an available function to read all images from an idx3 format to numpy arrays.

  2) convolution_manul.py --> In in this file we have built a code to run a convolution with a given/default filter/weight in a list of images. With the results, we can then run the code idx3_format.py to format all to idx3.


# Code Classifiers

* Our goal is to test our problem with different classifiers, in order to get an idea of how hard/accurate is the given problem. In this project we have different python codes for Classifiers:
# CNN Find Glasses in Faces

# Main Idea

The idea of this project is to develop a CNN to recognize glasses in Faces.

# Requirements

* Python3
* Tensorflow
* Sklearn
* Numpy
* Pandas
* Matplotlib
* Imageio
* Scipy

# Steps of our solution

1) Run a manual convolutional with a filter for edge detecter and save the new pictures. We also make the picture brighter where there are more details. This helps to highlight the glasses and also to reduce complexity

2) Convert all convoluted images to idx3 - Ubyte - This helps avoiding doing the same Data-Preprocessing many times

3) Feed the final result of convolved images to our models( perceptrons, CNN, ANN and others) to classify a picture as having glass or not.


# Data-Preprocessing

* In this project we have 2 different python codes for Data-Preprocess:

  1) idx3_format.py --> Used to convert a given list of images to idx3 format(MNIST format). We also have an available function to read all images from an idx3 format to numpy arrays.

  2) convolution_manul.py --> In in this file we have built a code to run a manual convolution with a given/default filter/weight in a list of images. With the results, we can then run the code idx3_format.py to format all to idx3. The idea of this program is to run a convolution of the image, to reduce details and keep the consistency. Feeding these new images to our model reduces the complexity of the problem.
  </br></br> Examples of output :
  <table border=1>
     <tr align='center'>
        <td>Original Image</td>                    
        <td>Convolved Image</td>                    
        <td>Label</td>                    
     </tr>
     <tr align='center' > 
        <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/conv_tests/original_0.png" width="350"                  title="hover text"></td>         
       <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/conv_tests/conv_0.png" width="350" title="hover        text"></td>
       <td>Not Using Glasses</td> 
     </tr>
  
    <tr align='center' > 
        <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/conv_tests/original_1.png" width="350"                title="hover text"></td>         
       <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/conv_tests/conv_1.png" width="350" title="hover        text"></td>
       <td>Using Glasses</td> 
     </tr>
  </table>
  
# Code Classifiers

* Our goal is to test our problem with different classifiers, in order to get an idea of how hard/accurate is the given problem. 

* In this project we have different python codes for Classifiers:

  1) full_perceptrons.py --> The idea is to test our model with only 1 layer(full layer of perceptrons), without hidden or convolution layers. Our goal is to get a better result than if a computer were just guessing.

  2) cnn_glasses.py --> We will use convolution layers, each one followed by a max pooling. At the end, we flatten the results, and give to a fully connected layer with one hidden layer. We will use a simplifcation of a restnet architecture.
 
# CNN Architecture
We started with a simple configurations and increased the "complexity" of our model step by step. 
</br>Our best resuts were achieved with the following configuration:
* 1 convolotion layers with 32 weights with filter size of 7 +
* 1 pooling +
* 1 convolotion layers with 64 weights with filter size of 7 +
* 1 pooling +
* 1 convolotion layers with 128 weights with filter size of 7 +
* 1 pooling +
* 1 convolotion layers with 256 weights with filter size of 7 +
* 1 pooling +
* 1 fully connected layer +
* drop of 0.50% +
* 1 output layer with 2 classes.


# Results

* The overal accuracy of our model is 98%.

# How to run our solution

* python3 cnn_glasses.py "Image path(absolute) to be tested"
* Output:
  * yes or no, for image with glasses or not.

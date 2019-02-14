# CNN Find Glasses in Faces

# Goal

The idea of this project is to develop a CNN to recognize glasses in Faces.

# Motivation
This project can be applied to help solve many social problems by using image classification. As a fact, the most active internet users are children and online safety issues regarding child protection is a dilemma. They are, unfortunately, subject to a number of threats online and virtual pornography is one of them. To prevent such obstacle in children’s childhood, we can use the power of ML to write a similar idea of project, to detect specific features in images. In this case, a suitable ML could detect nudity in images or videos and hide them from children by covering with clothes. 

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

* Our goal is to test our problem with different classifiers, in order to get an idea of how hard/difficult is the given problem. 

* In this project we have different python codes for Classifiers:

  1) full_perceptrons.py --> The idea is to test our model with only 1 layer(full layer of perceptrons), without hidden or convolution layers. Our goal is to get a better result than if a computer were just guessing.

  2) cnn_glasses.py --> We will use convolution layers, each one followed by a max pooling. At the end, we flatten the results, and give to a fully connected layer with one hidden layer. We will use a simplifcation of a restnet architecture.
 
# CNN Architecture
We started with a simple configurations and increased the "complexity" of our model step by step. 
</br>Our best resuts were achieved with the following configuration:
* 1 convolotion layer with 32 weights with filter size of 7 +
* 1 pooling +
* 1 convolotion layer with 64 weights with filter size of 7 +
* 1 pooling +
* 1 convolotion layer with 128 weights with filter size of 7 +
* 1 pooling +
* 1 convolotion layer with 256 weights with filter size of 7 +
* 1 pooling +
* 1 fully connected layer +
* dropout of 0.50% +
* 1 output layer with 2 classes.

# Inside Of CNN Layers
The ilustrations below is what out model was looking for in a image, when feeding an image with glasses. We can notice
that in some cases the glasses get very highlighted. In other cases, we humans cannot undestand/know what the filter/weight is looking for. However, in the last filter(7x7 picture) we can see in some cases a circular image with a high peak of pixels which may be a part of a glass.

</br>Layer 1 - 56 x 56 filtered images
  <table border=1>
     <tr align='center' > 
        <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_1_0.png" width="150"</td>         
       <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_1_1.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_1_2.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_1_3.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_1_5.png" width="150"</td>
     </tr>
    <tr align='center' >
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_1_18.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_1_19.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_1_25.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_1_26.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_1_29.png" width="150"</td>       
     </tr>
  </table>
  
  </br>Layer 2 - 28 x 28 filtered images
  <table border=1>
     <tr align='center' > 
        <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_2_0.png" width="150"</td>         
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_2_3.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_2_4.png" width="150"</td>
           <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_2_20.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_2_21.png" width="150"</td>
     </tr>
    <tr align='center' >
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_2_5.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_2_6.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_2_8.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_2_9.png" width="150"</td> 
           <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_2_17.png" width="150"</td>
     </tr>
     <tr align='center' >
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_2_22.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_2_25.png" width="150"</td>       
           <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_2_28.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_2_30.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_2_41.png" width="150"</td>
     </tr>
      <tr align='center' >
 </table>
 
  </br>Layer 3 - 14 x 14 filtered images
  <table border=1>
     <tr align='center' > 
        <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_3_0.png" width="150"</td>         
       <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_3_2.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_3_11.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_3_10.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_3_19.png" width="150"</td>
     </tr>
    <tr align='center' >
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_3_42.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_3_50.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_3_109.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_3_101.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_3_92.png" width="150"</td>       
     </tr>
</table>

  </br> Layer 4 - 7 x 7 filtered images
  <table border=1>
     <tr align='center' > 
        <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_4_12.png" width="150"</td>         
       <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_4_10.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_4_19.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_4_50.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_4_73.png" width="150"</td>
     </tr>
    <tr align='center' >
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_4_98.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_4_125.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_4_200.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_4_241.png" width="150"</td>
         <td><img src="https://github.com/thiagosantos1/CNN_Find_Glasses/blob/master/dataset/inside_conv/layer_4_224.png" width="150"</td>       
     </tr>
  </table>
  </table>


# Results

* The overal accuracy of our model is 98%. It was achied with the following configuration:
  1) Number of STEPS   = 50000
  2) Batch Size        = 40
  3) Learning Rate     = 0.00001
  4) Size Hidden Layer = 64
  5) Training Size     = 80%
  6) Test Size         = 80%

# How to run our solution

* python3 cnn_glasses.py "Image path(absolute) to be tested"
* Output:
  * yes or no, for image with glasses or not.

# Contributors
* M.Sc. Thiago Santos
* Dr. Geoff Exoo 

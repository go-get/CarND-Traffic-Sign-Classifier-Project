#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
   Used german traffic sign data provided downloaded from the course website. This gave input data in the exact model format, that works with sample LeNet grayscale code given as reference for number recognition. 
* Explore, summarize and visualize the data set
   The data set was single set, hence was split, 80% of it serving as training data, and 20% for testing. The images were 32x32 sized r,g,b images (32,32,3).  
* Design, train and test a model architecture
   The well established LeNet model was used to design our deep learning network. LeNet was configured, tuning to the problem at hand. 
* Use the model to make predictions on new images
   I achieved a accuracy of 95.5% on the validation set. 
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/atif-hussain/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the first,second,third code cell of the IPython notebook.  

Used german traffic sign data provided downloaded from the course website. This gave input data in the exact model format, that works with sample LeNet grayscale code given as reference for number recognition. 

The data set was single set, hence was split, 80% of it serving as training data, and 20% for testing. The images were 32x32 sized r,g,b images (32,32,3).  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?   27839 samples
* The size of test set is ?       6960 samples
* The shape of a traffic sign image is ?  (32, 32, 3)
* The number of unique classes/labels in the data set is ?  3072

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fourth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is randomly picked sample image from the training set. ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the sixth code cell of the IPython notebook.

The well established LeNet model was used to design our deep learning network. LeNet was configured, tuning to the problem at hand. 
Input layer: is now 5x5x3 color input, as color plays important role in traffic signs
Output layer: is now 43 size, as final classification can be any of 43 signs. 

For an effective Deep Learning Neural Network, LeNet model was used, but the dimensions were tuned to make it effective. 
We started with 


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the second code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set, taking 20% of total available data as validation set. 

My final training set had 27839 number of images. My validation set and test set had 6960 number of images.

The last code cells of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because ... To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x10 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x10 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten       	    | output  400      								|
| Fully connected 		| 120        									|
| Fully connected 		|  80        									|
| Fully connected 		|  43        									|
| Softmax   			|  ..        									|
|						|												|
|						|												|
 

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used the standard 

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?    95.5%
* validation set accuracy of ?  95.5%
* test set accuracy of ?        

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
    LeNet model architecture
* Why did you believe it would be relevant to the traffic sign application?
    It was object detection, and it'd worked well for number recognition
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    The Accuracy came out strong on training set, without adding complexities. So it is not overfitted and accuracy on validation and test set will also be good. 
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![stop][stop.jpeg] ![U-turn][testSet/U-turn.jpeg] ![yield][testSet/yield.jpeg]  
![100kmph][100kmph.jpg] ![slippery][slippery.jpg]

Could not classify because unable to call the python model to run on these images. 


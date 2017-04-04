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

####0. Sourcing of the data set. 
This was done in the first code cell of the IPython notebook.  

Used german traffic sign data provided downloaded from the course website. This gave input data in the exact model format, that works with sample LeNet grayscale code given as reference for number recognition. 

The data set was a single set, hence was split - 80% of it serving as training data, and 20% for testing. See second code cell. 

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell of the IPython notebook.  

The images were 32x32 sized r,g,b images (32,32,3).  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?   27839 samples
* The size of test set is ?       6960 samples
* The shape of a traffic sign image is ?  (32, 32, 3)
* The number of unique classes/labels in the data set is ?  43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fourth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. 
  A randomly picked sample image from the training set. ...
  ![alt text][image1]


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the sixth code cell of the IPython notebook.

The well established LeNet model was used to design our deep learning network. LeNet was configured, tuning to the problem at hand. 
Input layer: is now 5x5x3 color input, as color plays important role in traffic signs
Output layer: is now 43 size, as final classification can be any of 43 signs. 

For an effective Deep Learning Neural Network, LeNet model was used, but the dimensions were tuned to make it effective. Also, began by randomizing inputs to prevent ordering dependent fits. 

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the second code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set, taking 20% of total available data as validation set. 

My final training set had 27839 number of images. My validation set and test set had 6960 number of images. This was the Distribution of training inputs across classes.
  inputs_per_class= [ 142 1594 1607 1019 1377 1323  307 1026  996 1072 1456  948 1507 1515  544
  439  282  799  861  141  231  210  264  352  194 1059  435  162  382  197
  323  556  175  483  304  869  262  143 1496  215  238  174  160]
These were found to be well represented across in the training data, with each class having a minimum of over hundred test inputs. 
  ![alt text][image2]

A quick dirty validation of the model for applicability was done, by running it for 3 EPOCHS. 
EPOCH	 Training Accuracy	Validation Accuracy 
 1  		 0.654  			0.641
 2  		 0.858  			0.839
 3  		 0.904  			0.886
Using AdamOptimizer for faster training, the Model started with decent accuracy, and began improving quite quickly, reaching ~90% accuracy in 3 epochs. So it was run for a basic 10 epochs to go well beyond the goal of 93% accuracy.  
The final test data was used only after the model was completely trained. 

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

While tuning the dimensions of LeNet model, dimensions were so chosen, and care was taken to limit classes at each stage to control computational complexity and prevent overfitting. 

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth, tenth cells of the ipython notebook. 

To train the model, I used the standard AdamOptimizer that performs better than StochasticGradientDescent. Minimize op was used to call compute_gradients() and apply_gradients() in sequence. Recommended Learning rate, hyperparameters and batch size, were taken from the MNIST computer vision exercise using LeNet. The model was trialed using 3 EPOCHs, and found to fit well on both training & validation sets. It was extended to basic 10 EPOCHs to go beyond the goal of 93% accuracy. It was noted that even for this small no of EPOCHs, the model accuracy had started deteriorating. In my training runs, it reached test and validation accuracies of 97% and 94%, and plateaued there. 

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?    0.980
* validation set accuracy of ?  0.956
* test set accuracy of ?        0.876

If an iterative approach was chosen: With a strong start using LeNet for computer vision on MNIST, and using gradual reducing of classes acrosses Deep learning stages, no iterations were found necessary. 
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?new_predictions = session.run(prediction, feed_dict={input_ph: new_input})


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

![20][testSet/20.png] ![80][testSet/80.png] ![caution][testSet/caution.png]  
![keepleft][testSet/keepleft.png] ![roadwork][testSet/roadwork.png]

The model accuracy for images on the web was only 20%. 

Some of the low accuracy on sample images is because of the skewness of training data towards few labels. If a particular label is rare in the training set, then the model can keep training the model to high accuracies without ever detecting rare labels. In case of our samples, the model wrongly predicted same class for 3 of 5 cases. 
Here's a look at the 5 different images, and their abundance in training-cum-validation set, along with prediction accuracy (sorted). 
|'Image' 		| Actual label	|Test Abundance	|Test rank#	|Detection rank#|	Probability	|
|'80.png'		|	 5 			|	 0.048 		|	 8th	|		  1 	|	0.065 		|
|'roadwork.png'	|	25 			|	 0.038 		|	10th	|		  8 	|	0.032 		|
|'caution.png'	|	18 			|	 0.031 		|	16th	|		 13 	|	0.026		|
|'keepleft.png'	|	39 			|	 0.008 		|	33rd	|		 11 	|	0.030 		|  
|'20.png'		|	 0 			|	 0.005 		|	42nd	|		 43 	|	0.00 		|

The model predicted only one of five signs correctly, the 80 speed limit sign. The correct label was outside the top5 for all other samples, but within top 13 (of 43) for 3 more signs, 2 of which along with detected sign had high prevalence in test data. Model prediction was completely useless for label classes that were rare in the training+validation set. 
To make the model more accurate across different classes, we can increase the share of underrepresented classes in the training set. One option to do this is to generate additional data for these classes, or just reuse some of these inputs to increase their weight. 
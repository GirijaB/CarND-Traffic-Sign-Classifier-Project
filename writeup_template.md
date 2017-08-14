#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
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

You're reading it! and here is a link to my [project code](https://github.com/GirijaB/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
I downloaded the data set as suggested in the courseware and the images used for training and validating were 32x32 sized r,g,b images (32,32,3).

signs data set:

* The size of training set is ? 31367
* The size of the validation set is ? 7842
* The size of test set is ? 12630
* The shape of a traffic sign image is ?  (32, 32, 3)
* The number of unique classes/labels in the data set is ? 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

The code to visualize the dataset is in the 3rd and 4th cells. In the below graph the traffic sign labels (for training data) are shown in the x-axis and the count of photos are shown in the Y-axis. As can be seen from the graph, dataset is not balanced.

![alt text](https://github.com/GirijaB/CarND-Traffic-Sign-Classifier-Project/blob/master/image1.png)

Visualisation of some random images:-

![alt text](https://github.com/GirijaB/CarND-Traffic-Sign-Classifier-Project/blob/master/image2.png)


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The code to preprocess data is in cells 3-4-5 of the IPython notebook .

The well established LeNet model is used to design our deep learning network. LeNet was configured, tuning to the problem at hand. Input layer: is 5x5x3 color input, as color plays important role in traffic signs Output layer: is 43 size, as final classification can be any of 43 signs.

For an effective Deep Learning Neural Network, LeNet model was used, but the dimensions were tuned to make it effective. Also, began by randomizing inputs to prevent ordering dependent fits.

The code for splitting the data into training and validation sets is contained in the second cell of the IPython notebook.

To cross validate my model, I randomly split the training data into a training set and validation set, taking 20% of total available data as validation set.

My final training set had 31367 number of images. My validation set and test set had 7842 number of images. This was the Distribution of training inputs across classes. inputs_per_class= [ 169 1781 1799 1134 1578 1485  332 1143 1147 1189 1619 1040 1691 1730  614  513  329  884  982  176  286  261  313  411  203 1181  467  193  415  212  349  623  201  576  348  983  317  161 1627  235  289  186  195]
These were found to be well represented across in the training data, with each class having a minimum of over hundred test inputs.

A quick dirty validation of the model for applicability was done, by running it for 3 EPOCHS. EPOCH Training Accuracy Validation Accuracy 0.551 0.774 2 0.850 0.882 0.918 0.942 0.962 and 0.963 Using AdamOptimizer for faster training, the Model started with decent accuracy, and began improving quite quickly, reaching ~90% accuracy in 3 epochs. So it was run for a basic 10 epochs to go well beyond the goal of 93% accuracy.
The final test data was used only after the model was completely trained.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x10 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x10 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16     									|
| Max pooling	      	|2x2 stride, outputs 5x5x16				|
| Flatten 	|output 400 |
| Fully connected		|120      									|
| Fully connected		|80      								 	|
| Fully connected		|43     								  	|
| Softmax				| etc.        									|
|						|												|
|						|												|
 
While tuning the dimensions of LeNet model, dimensions were so chosen, and care was taken to limit classes at each stage to control computational complexity and prevent overfitting.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth, tenth cells of the ipython notebook.

To train the model, I used the standard AdamOptimizer that performs better than StochasticGradientDescent. Minimize op was used to call compute_gradients() and apply_gradients() in sequence. Recommended Learning rate, hyperparameters and batch size, were taken from the MNIST computer vision exercise using LeNet. The model was trialed using 3 EPOCHs, and found to fit well on both training & validation sets. It was extended to basic 10 EPOCHs to go beyond the goal of 93% accuracy. It was noted that even for this small no of EPOCHs, the model accuracy had started deteriorating. In my training runs, it reached test and validation accuracies of 97% and 94%, and plateaued there.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:

training set accuracy of ? 0.963
validation set accuracy of ? 0.936
test set accuracy of ? 0.854

If an iterative approach was chosen: With a strong start using LeNet for computer vision on MNIST, and using gradual reducing of classes acrosses Deep learning stages, no iterations were found necessary.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text](https://github.com/GirijaB/CarND-Traffic-Sign-Classifier-Project/blob/master/newImages/60_speed.jpg)
![alt text](https://github.com/GirijaB/CarND-Traffic-Sign-Classifier-Project/blob/master/newImages/STOP_sign.png)
![alt text](https://github.com/GirijaB/CarND-Traffic-Sign-Classifier-Project/blob/master/newImages/keep_left.png)
![alt text](https://github.com/GirijaB/CarND-Traffic-Sign-Classifier-Project/blob/master/newImages/road_work_ahead.png)
![alt text](https://github.com/GirijaB/CarND-Traffic-Sign-Classifier-Project/blob/master/newImages/school_zone.png)



The model accuracy for images on the web was only 20%.

Some of the low accuracy on sample images is because of the skewness of training data towards few labels. If a particular label is rare in the training set, then the model can keep training the model to high accuracies without ever detecting rare labels. In case of our samples, the model wrongly predicted same class for 3 of 5 cases.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| acceptable
| keep_left    			| keep_left 										|aceptable
| 60_speed					| 60_speed											|acceptable
|road_work_ahead      		|road_work_ahead 			 				|not good
| school_zone		| school_zone   							|not good




####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.
Model prediction was completely useless for label classes that were rare in the training+validation set. To make the model more accurate across different classes, we can increase the share of underrepresented classes in the training set. One option to do this is to generate additional data for these classes, or just reuse some of these inputs to increase their weight.

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.But model accuracy was only 20% on web. 
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



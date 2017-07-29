#**Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration


###Design and Test a Model Architecture

####1. Image Preparation


####2. Model Architecture

####3. Model Training

####4. Solution Approach

To determine direction for model architecture, I looked for structures that have performed well during image classification challenges.  To avoid too much complexity with the likes of GoogLeNet and ResNet, and a fear of long training times with AlexNet and , I stayed with LeNet but made modifications:
1. The first two convolution layers in [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) perform include local contrast normalizations (LCN) to aid in generalization which helped improve error rates.  I added these steps to my network also.
2. I added dropout to the two fully connected layers to assist with ovrfitting.  I wanted to quickly test if these changes had improvements, therefore, I updated the LeNet notebook with my changes and re-tested.  The [results](LeNet-Lab-Solution.html) yielded a Test Accuracy of 0.990.
3. I uncovered a jem of an article [14 Design Patterns To Improve Your Convolutional Neural Networks](http://www.topbots.com/14-design-patterns-improve-convolutional-neural-network-cnn-architecture/) with tips for customizing my CNN.


###Test a Model on New Images


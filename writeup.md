# **Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

This Writeup will address these project goals as seen in my [project code](https://github.com/Merberg/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

[//]: # (Image References)

[iAnimal]: ./examples/AnimalCrossing.png "Visually Close"
[iPreproc]: ./examples/Preprocessed.png "Preprocessing"
[iBarh]: ./examples/TrainingDataLabelsBarh.png "Training Label Quantities"
[iTestBarh]: ./examples/TestDataBarh.png "Testing Label Quantities"
[iValAcc]: ./examples/ValidationAccuracy.png "Validation Accuracy.png"
[iTSCNN]: ./examples/TrafficSignCNN.PNG "Traffic Sign CNN"
[iSign1]: ./new_traffic_signs/Child1.jpg "Traffic Sign 1"
[iSign2]: ./new_traffic_signs/Child2.jpg "Traffic Sign 2"
[iSign3]: ./new_traffic_signs/NoEntry_bike.jpg "Traffic Sign 3"
[iSign4]: ./new_traffic_signs/roundabout.jpg "Traffic Sign 4"
[iSign5]: ./new_traffic_signs/slippery-road.jpg "Traffic Sign 5"
[iSign6]: ./new_traffic_signs/Yield.jpg "Traffic Sign 6"
[iProb1]: ./examples/Class28.png "Traffic Sign 1 Probabilities"
[iProb2]: ./examples/2ndClass28.png "Traffic Sign 2 Probabilities"
[iProb3]: ./examples/Class17.png "Traffic Sign 3 Probabilities"
[iProb4]: ./examples/Class40.png "Traffic Sign 4 Probabilities"
[iProb5]: ./examples/Class23.png "Traffic Sign 5 Probabilities"
[iProb6]: ./examples/Class13.png "Traffic Sign 6 Probabilities"


### Data Set Summary & Exploration
Using python methods, I programmatically calculated data set statistics:
- Number of training examples = 34799
- Number of validation examples = 4410
- Number of testing examples = 12630
- Image data shape = 32x32x3
- Number of classes = 43

The number of examples per class provides insight into a potential training problem; the training data set is skued towards common traffic signs, which could drop the accuracy for the other signs.  Those examples with fewer training images need to be supplimented with additional ones to even the playing field.

![Bar Graph of Examples per Class][iBarh]

Distribution does not tell the entire story as even human processing could be fooled by some images at first glance:

![Wild Animal vs Double Curve][iAnimal]

This highlights the need for image pre-processing.

### Design and Test a Model Architecture
#### 1. Image Preparation
My image pre-processing followed three steps:
1. Convert images to grayscale to help with light and color inequalities.
2. Normalize the images to try for a mean of zero and equal variance. (Actuals for the training data set: Mean = -0.37, Std = 0.51)
3. Suppliment the classes with fewer than one standard deviation of image examples with warped images.  For the training data set, this resulted in an additional 7739 examples.

![Preprocessed][iPreproc]

#### 2. Model Architecture
The following represents my final model:

![LeNet + ELU + LCN][iTSCNN]

*Graph created by implementing the network layers in a MATLAB script*

| | Layer | Description |
|-|--------|--------|
| 0 | Input | 32x32x1 Pre-processed Image |
| 1 | Convolution 5x5 | 1x1 stride, valid padding, output 28x28x8
| 1a | Activation | ELU |
| 1b | Normalization | depth_radius 2, alpha 2e-05, beta 0.75, bias 1.0
| 1c | Max Pooling | 2x2 stride, output 14x14x8
| 2 | Convolution 5x5 | 1x1 stride, valid padding, output 10x10x16
| 2a | Activation | ELU |
| 2b | Normalization | depth_radius 2, alpha 2e-05, beta 0.75, bias 1.0
| 2c | Max Pooling | 2x2 stride, output 5x5x16
| 2d | Flatten | output 400x1
| 3 | Fully Connected | output 120x1
| 3a | Activation | ELU |
| 3b | Dropout | |
| 4 | Fully Connected | output 84x1
| 4a | Activation | ELU |
| 4b | Dropout | |
| 5 | Fully Connected | output 42x1
| 5a | Activation | softmax |

#### 3. Model Training
During training with an Adam optimizer, I discovered that each network change resulted in a hyperparameter adjustment. To simplify things, I did the majority of my testing with a higher learning rate to monitor how Epochs and Batch Size impact accuracy.  Once I settled in on those values, I followed the "keep calm and drop your learning rate" approach.  One interesting note is that when I changed ReLUs to ELUs with an Epoch of 150, my accuracy decended by 2% when complete.  Dropping the number of Epochs produced the results below:

![Final Validation Accuracy][iValAcc]

```
LEARNING_RATE = 0.0005
EPOCHS = 100
BATCH_SIZE = 75
```

#### 4. Solution Approach

To determine direction for model architecture, I looked for structures that have performed well during image classification challenges.  To avoid too much complexity with the likes of GoogLeNet and ResNet, and a fear of long training times with AlexNet, I stayed with LeNet but made modifications:
1. The first two convolution layers in [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) include local contrast normalizations (LCN) to aid in generalization which helped improve error rates.  I added these steps to my network also.
2. I added dropout to the two fully connected layers to assist with ovrfitting.  I wanted to quickly test if these two changes had improvements, therefore, I updated the LeNet notebook with my changes and re-tested.  The [results](LeNet-Lab-Solution.html) yielded a Test Accuracy of 0.990.
3. I uncovered a jem of an article [14 Design Patterns To Improve Your Convolutional Neural Networks](http://www.topbots.com/14-design-patterns-improve-convolutional-neural-network-cnn-architecture/) with tips for customizing my CNN.  One quick change was replacing the ReLUs with ELUs.


### Test a Model on New Images

![Sign 1][iSign1]![Sign 2][iSign2]![Sign 3][iSign3]![Sign 4][iSign4]![Sign 5][iSign5]![Sign 6][iSign6]

My additional German traffic signs, pictured above, possed some challenges for my network due to disparities in training examples for some of these classes.

|  | Image	|  | Prediction |
|:--:|:-----------------:|:--:|:----:|
| 28 | Children crossing | 14 | Stop |
| 28 | Children crossing | 36 | Go straight or right |
| 17 | No entry | 7 | Speed limit (100km/h) |
| 40 | Roundabout mandatory | 40 | Roundabout mandatory |
| 23 | Slippery road | 23 | Slippery road |
| 13 | Yield | 13 |Yield |
```
New Image Accuracy = 50.0%
Test Accuracy = 0.948
```
The test data shows similar bias as the training data, which helps explains its higher accuracy vs the new signs:

![Testing Data Labels][iTestBarh]

The softmax probabilities show that the network had a slim chance of correctly identifying the failing classes as they did not appear in the top five.  On the other end of the spectrum, the passing signs hit above an 88%, even reaching 100% probabilities for the Slippery road and Yield.

*Failing Classes:*

![Sign 1][iSign1]
![Sign 1 Probabilities][iProb1]

![Sign 2][iSign2]
![Sign 2 Probabilities][iProb2]

![Sign 3][iSign3]
![Sign 3 Probabilities][iProb3]


*Passing Classes:*

![Sign 4][iSign4]
![Sign 4 Probabilities][iProb4]

![Sign 5][iSign5]
![Sign 5 Probabilities][iProb5]

![Sign 6][iSign6]
![Sign 6 Probabilities][iProb6]






**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 containing the video demonstrating the autonomous operation of the car using the trained model
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I started from the LeNet architecture, and modified the filter depths to start with. After that I added a few more layers.
Overall the model consists of;
* cropping layer to crop the top part of images
* lambda layer used for scaling and centering of the image pixel values 
* four conv2d layers, each followed by a 2x2 strided maxpooling layer
* a flatten layer
* 5 dense or fully connected layers

####2. Attempts to reduce overfitting in the model
To check if the model was overfitting, I monitored the training loss and validaton loss as the epochs progressed. I did not see any signs of overfitting.

To further improve the model, I increased the size of the training set two times, each time adding more data to the training and validation sets.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (2-3 laps), recovering from the left and right sides of the road, and also did one lap of driving in the reverse direction.  

Further, I also used all the three camera images, i.e. left, center, and right images. I corrected the steering angle corresponding to left and right images. 

The training set was finally augmented by adding flipped image for each image in the set. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a good enough model, and then fine tune it by adding more data and correspondingly increasing the parameters, such as adding more layers or increasing the depth of filters.

My first step was to use a convolution neural network model similar to the LeNet architecture. I thought this model might be appropriate because we have used this as baseline architecture for different image classification tasks, such as digits classification and traffic sign classification. Further, this is simple enough and trains reasonably quickly. Also, it is small enough such that the risk of overfitting is low.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

The original model wasn't able to keep the car on track for very long. I then increased the data set size as mentioned in previous section and for each increase in data set size, the total number of model parameters were increased.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road for multiple laps.

####2. Final Model Architecture

The final model architecture (model.py lines 91-111) consisted of the following layers and sizes:

* cropping layer to crop the top part of images
* lambda layer used for scaling and centering of the image pixel values 
* four conv2d layers (sizes - 5x5x6, 5x5x16, 3x3x32, 3x3x48), each followed by a 2x2 strided maxpooling layer
* a flatten layer
* 5 dense or fully connected layers (sizes - 1164, 100, 50, 10, 1)

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to 

Then I recorded one lap of driving in the reverse direction.

To augment the data sat, I also flipped images and angles thinking that this would help with training the model to steer in the right direction, as most of the turns were towards the left.

After the collection process, I had 6417 number of data points. Using all the three images, i.e. left, center, and right, and also augmenting the images using flipping, brings the total number of images to 6417 * 3 * 2 = 38502.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by convergence of the validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.

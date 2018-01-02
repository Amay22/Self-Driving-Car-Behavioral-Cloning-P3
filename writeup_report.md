# WRITEUP 

# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image3]: ./trainig.png "Model Training"

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* output_video.mp4 a video output of the lap that the car took on track 1 using my model.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model I used is very similar to the one explained in the lectures and the suggested paper https://arxiv.org/pdf/1608.01230.pdf. The entire design consists of 3 Convolutions and 2 fully connected layer. The model follows a sequential design that I have described below. Look at the image below for further clarification.

* Normalized the data using `Lambda x: (x / 127.5) - 1.0`.
* Cropped out the top 75 pixels that includes the sky and the bottom 25 pixels which includes the car hood.
* 3 Convolition are applied one after another. `Convolution2D` funtion is used. All the convolutions' have border_mode set to same and the filter size started from 16 and was multiplied by 2 in every succeeding convolutions. The subsampling is taken as (4,4) in the 1st convolution and then as (2,2) in the other 2. 
* Every `Convolution2D` is followed by `ELU()` which is an Exponential Linear Units. It speeds up learning in deep neural networks and leads to higher classification accuracies. ELUs have improved learning characteristics compared to the units with other activation functions.
* 2 Fully connected layers are used after the 3 Convolutional layers. The the output from the 3 Convolutions is first flattened using `Flatten()` then Densed using `Dense()` function it outputs a shape of 512 and 1 in the second one which is very aggressive approach.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers before the fully connected layers to reduce overfitting the first uses 0.2 probability and the second one uses 0.5 probability. I reckones 0.1 was not enough and the once the network passes through one fully connected layer that squeezes (denses) the output into 512 I can use 0.5 in the next one as the output of the next fully connected layer is 1.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. I used the common prescribed learning rate of 0.001.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I tried recording the data, driving the car in the center of the road was easy but getting the car off-course and getting it back on was not and I don't think I stopped recording correctly. y initial training set with the car in the center went into the lake 4 times and after adding a reverse lap to my training data it still went into the water. I then used the sample data provided and it ended up making the turn properly but then it went right into the wall and did not reverse at all it was still accelerating and I added some samples to the provided data that included reversing the car. That's when my model started to perform somewhat. I think looked into improving my model architecture which was only 2 convolution layers and 2 Fully connected layer and I started improving the sample generation. The description of the model is above. The training set provided in the lecture is quite good enough.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The model is described above. I started of by splitting the data into 15% Validation data and 85% Training data. The provided csv file had 8037 and the 85% gave me a solid 6830 left, center, right images alongwith steering angle. I picked a `batch_size` of `64` which made the most sense based on the length of the data. The 64 data set was generated from the completed data to be pulled into the `fit_generator()` function.

I picked the 1 image from the left, center, right in a row at *random* and based on the image that I had picked I corrected the steering angle for left image I added -0.25 and for right image I added 0.25 and for the center image left the steering angle untouched. 

I *augmented* the image at *random* by flipping the image to the right side horizontally as suggested in the lecture and the negated the value of the steering angle alongwith it. 

A python genertor has been created to render these values out to the `fit_generator()` function.

#### 2. Final Model Architecture

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Training Process

I started the training process with 2 epochs and then tried to increase the accuracy by changing the model. The multiple `ELU()` methods in between helped. After adding the 3rd Convolution layer I was able to get the car to drive in like 5 epochs. I then added added the second Dropout and increased the epoch to 7. I then added the EarlyStopping Callback function which is pretty handy when as it stops the extra epochs when no accuracy is achieved. I have number of epochs as 10 but the EarlyStopping method normally stops the model execution in the 8th cycle.  

Training Model visualization:

![alt text][image2]
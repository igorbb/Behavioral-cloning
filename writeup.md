# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center]: ./sample.jpg "center"
[flip]: ./flip_out.jpg "flipped"
[mask]: ./mask_out.jpg "Recovery Image"
[brightness]: ./brightness_out.jpg "Recovery Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* Model.ipynb containing the script to create and train the model
* mask.jpg the mask used as pre-processing for out model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md or writeup.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The Model.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains rough annotations describing the code actions.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x11 filter sizes and depths between 32 and 128.

The model includes RELU layers to introduce nonlinearity in the convolution layers.
The model also includes tanh to introduce nonlinearity at the fully connected layer.

#### 2. Attempts to reduce overfitting in the model

The model contains L2 weight regularization for every convolution in order to reduce over-fitting (function `conv2d_bn`).
The model also two dropout layers to further reduce over-fitting.

The model was trained and validated on different data sets to ensure that the model was not over-fitting (as shown in the notebook cells [10 and 11]). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (notebook cell 8).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
I followed recommendations and tried to keep driving at the center.
The side cameras images are also used as augmentation technique; they are employed to teach  the car to recover for when driving away from the center of the road.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was iterative.

My first step was to use a convolution neural network model similar to the one I have used in my last project.
No signs of over-fitting were noticed. Both Validation and Training  sets had small mean squared error.
The results were satisfactory and the model could be used to drive  the car through the first track autonomously.

I also experimented by making the network that would also do regression for throttle and speed.
The results achieved with it were comparable but not better. Thus I went back to estimate solely the angle.

I decided to change the network to my final model because the model was having a hard time generalizing to the second track.
With larger convolutions filters, and the use of data augmentation the card started performing much better.

At the end of this process, the vehicle is able to drive autonomously around the both tracks without leaving the road.

#### 2. Final Model Architecture

The final model architect consisted of a convolution neural network with the following layers.
Each `Convolution2D` has 3x11 kernel size and employs `ReLu` activation.

The Dense layers have `tanh` activation except for the last one, which has no activation.

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input (InputLayer)               (None, 160, 320, 3)   0                                            
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 100, 320, 3)   0           input[0][0]                      
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 98, 310, 32)   3200        cropping2d_1[0][0]               
____________________________________________________________________________________________________
batchnormalization_1 (BatchNorma (None, 98, 310, 32)   128         convolution2d_1[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 49, 155, 32)   0           batchnormalization_1[0][0]       
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 47, 145, 64)   67648       maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
batchnormalization_2 (BatchNorma (None, 47, 145, 64)   256         convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 23, 72, 64)    0           batchnormalization_2[0][0]       
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 21, 62, 128)   270464      maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
batchnormalization_3 (BatchNorma (None, 21, 62, 128)   512         convolution2d_3[0][0]            
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 10, 31, 128)   0           batchnormalization_3[0][0]       
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 8, 21, 32)     135200      maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
batchnormalization_4 (BatchNorma (None, 8, 21, 32)     128         convolution2d_4[0][0]            
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 4, 10, 32)     0           batchnormalization_4[0][0]       
____________________________________________________________________________________________________
Flatten (Flatten)                (None, 1280)          0           maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1280)          0           Flatten[0][0]                    
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 256)           327936      dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 256)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 128)           32896       dropout_2[0][0]                  
____________________________________________________________________________________________________
Angle (Dense)                    (None, 1)             129         dense_2[0][0]                    
====================================================================================================
Total params: 838,497
Trainable params: 837,985
Non-trainable params: 512
____________________________________________________________________________________________________
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 4 laps on track one using center lane driving.
The first two laps were going on forward, and the final two the car did the reverse path.

Here is an example image of center lane driving:

![center Image][center]

Then I recorded few lap on the second track to add more points in our dataset.

To further augment the data we randomly either flip an image, do nothing to the image (identity operation), or change the image brightness.

The idea was to simulate different rode curves with image flipping. To this end we also invert the angles used in regression.

![flip image][flip]

To random simulate different lighting conditions, I converted the image to the HSV color space and randomly
modulate the brightness channel. The final image comes by inverting the HSV image back to RGB colorspace.

![brightness][brightness]


All these images would be masked before fed to the the model.
The mask is to set a region of interest in which the network will use to learn the regression model.


![masked][mask]

We also use the side cameras to augment the dataset by 3. This way
After the collection process, I had 8263 number of data points.
With the data augmentation we consider to have 3 times more images, totaling 24789 images.
In reality the total number of unique images used is much bigger, this due to randomness in the data augmentation


I finally randomly shuffled the data set and put 15% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by matching losses plateaus. I used an adam optimizer so that manually training the learning rate wasn't necessary.

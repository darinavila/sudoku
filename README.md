# sudoku

Readme Instruction:
> https://guides.github.com/features/mastering-markdown/

# SETUP & INSTRUCTIONS


# COMPUTER VISION OVERVIEW
Computer vision is the field concerning the ability of a computer to process photo or video input that it receives from a visual sensor. This is accomplished by converting an image into an array of tuples of three pixels with values between 0 and 255 (representing the RGB color system), and representing a video as a set of images which continually change to reflect the current frame. This array can then be manipulated for a variety of purposes, including motion detection, object classification, filter application, among many others. The use of Python’s opencv library for computer vision which operates on top of Python’s numpy library for linear algebra makes computer vision tasks much simpler than they otherwise would be. 


# SUDOKU PROJECT OVERVIEW
The code I was tasked to reverse engineer is a program which takes a video of someone holding up a sudoku puzzle as the input, and prints out the exact same video with the entire grid filled in with the solved puzzle. This is a complex task, but it can be thought of in three manageable steps. 

From the raw video input, isolate the portion of the frame containing the sudoku grid.
From the grid image, locate and identify the numbers which are currently printed on the board and input them into a matrix.
Use a known Sudoku algorithm, solve this matrix and print out the numerical solutions into their respective spots on the sudoku grid.

Each of these steps are critical to this particular use case of opencv, but in order to modify this code to fit a different object classification use case as you had mentioned step 2 (and, to a lesser degree, step 1) is the most critical to understand. 



# SUDOKU GRID DETECTION

The first step is to find the contour, or region of the frame which contains the sudoku grid. Below is the code I used to draw this contour. (NOTE: While the creator’s github was consulted for ideas and certain opencv methods, all code here is original)

import cv2, time
import numpy as np
video = cv2.VideoCapture(0)

x = 1

while True:
    check, frame = video.read()
    frame = frame
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame,(5,5),0)
    frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
    image, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
    maxA = -1
    maxC = None
    secA = -1
    secC = None
   
    for f in contours:
        if cv2.contourArea(f) > maxA:
            maxA = cv2.contourArea(f)
            maxC = f
        elif cv2.contourArea(f) > secA:
            secA = cv2.contourArea(f)
            secC = f
   
   
    big = [1]
    big[0] = secC
    print(secC)
   
    check, frame2 = video.read()
    final = cv2.drawContours(frame2, big, -1, (0,0,255),1)
    cv2.imshow('final',final)
    key = cv2.waitKey(1)
    if key == ord('x'):
        break
video.release()
cv2.destroyAllWindows



![mypic](images/grid_grab.png)











In short, I set up a loop to constantly be grabbing new camera frames, thus creating a real time video. Then, I took each individual frame and converted it to a black and white image to make image processing possible. Then I applied Gaussian blurring in order to remove image noise, and adaptive Gaussian thresholding to mark tangible “edges” in the photo. I then found all of the contours, which are essentially just the boundary of closed regions formed from these edges. Because the entire frame was being counted as one big contour, I wrote code to grab the second biggest contour, and then to draw this back on the original frame.



After this I grabbed the polygon encompassed by the grid with the lines

poly = cv2.approxPolyDP(secC, 1, True)
cv2.fillConvexPoly(final, poly, (0,255,255), 1)
Which yields the image below


![mypic](images/grid_poly.png)

After this, the rotated rectangle of the grid must be mapped to an upright rectangle in order for number classification to occur, so we have to use the warpperspective function to accomplish this. In a use case such as the one you described to me, this may not be necessary, but some form of rotating an object to get the proper perspective is indeed important in object classification.


# NUMBER DETECTION (THIS IS THE CRITICAL STEP FOR AR USE CASE)

There are some minutiae in the beginning of this phase of the project as the grid needs to be sliced into its different boxes and the lines between the numbers ignored, but this portion of the code is specific just to sudoku and won’t be necessary for other use cases. I have pasted the code in this section below, and will offer a brief explanation of its parts, but what’s important is what happens after these boxes have been compartmentalized.  Each image of a number is “looked” at by the computer, input into a convolutional neural network, and output as a digit between 0-9. The beauty of this process is that the fact that this is happening in real time is no obstacle, as each individual frame is looked at separately, and its grid located and numbers classified instantaneously.  Each still image will run images through the neural network and receive numbers. The accuracy of this type of numerical classification is over 99%, and this is owed in large part to the power of the convolutional neural network. 


# CONVOLUTIONAL NEURAL NETWORK THEORY

Convolutional neural networks take in a tensor (for our purposes, this is a numpy matrix of pixels) of values, which is then convolved with a filter, or simply changed according to some function involving both the input image and the filter, where the filter is a matrix of numbers which can change an image when the two are convolved. This convolution of the image and the filter is then sent forwards as a signal through the network if it passes a certain activation function, and this continues at each level of the network until the final layer is reached and the image is sorted into one or multiple of however many categories there are for the image. While the network is training, if the image is sorted incorrectly, this error will be sent back through the layers and the filters will be updated via backpropagation just as in a normal neural network, and this is how the network “learns”.  What sets a CNN apart from other neural networks is the fact that its layers of neurons are fully connected, which gives it the ability to understand more complex visual patterns. This technology should be used in creating a real-time image classifier if maximum accuracy is important.

# CNN CODE IN PYTHON
CNN’s can be quickly and easily implemented with the Keras syntax on top of the Tensorflow 2.0 library. There are countless examples online of simple implementations for digit classification. The github example uses a labeled dataset of printed numbers for training, but I will show here an example of a quick implementation of handwritten digit classification with the famous MNIST dataset, and explain each part in detail. I got this example from https://keras.io/examples/mnist_cnn/. 

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

Imports the necessary libraries

batch_size = 128
num_classes = 10
epochs = 12


The batch size is how many entries of data it trains on at once, the num_classes is the number of output categories (10 for 10 digits 0-9) and the number of epochs is how many times it trains on a batch 

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

This step imports the MNIST labeled data as x and y for the training and testing set

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()

The above step creates a sequential, or layer based neural network that can now be filled with layers or neurons

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))

The above two steps add two convolutional layers with the relu activation function

model.add(MaxPooling2D(pool_size=(2, 2)))

The above step adds a pooling layer whose purpose is to condense the output of a layer of neurons into one neuron for computational purposes

model.add(Dropout(0.25))

The above step drops a certain proportion of the neurons to be trained in each iteration in order to prevent overfitting (which CNNs are prone to due to their fully connected nature)

model.add(Flatten())

The above step flattens the output of the previous layer’s neurons to the proper number of channels for the next layer

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

The above two steps add another layer and drop a different proportion of neurons

model.add(Dense(num_classes, activation='softmax'))

Finally we have out output layer which has 10 output categories (1 for each digit) and uses the softmax activation function to probabilistically determine which digit an image is supposed to represent

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

The above two steps compile the model where we have chosen categorical cross entropy as our cost function (as is standard for most categorical neural networks) and picked a gradient descent optimizer Adadelta.

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

The above steps calculate and print the error in the model, which reaches above 99% by the 12th epoch in a relatively short amount of time


# MODIFICATIONS
So how can code like this be adapted for a different type of object classification? Most of the body of this code can remain the same. We will continue to need a sequential model with convolutional layers and then a flattening into additional “normal” dense hidden layers, all with the efficient relu activation. Depending on the size of what we’re doing it might be very helpful or even necessary to include pooling layers, and dropping a proportion of the neurons is always important to prevent overfitting. What will change then will of course be the labeled dataset we are using to train the network, and perhaps also the size of the image we enter into the network. The number of output modes which the final softmax function will sort images into is again entirely dependent on the use case, and how many categories of objects the network is meant to classify. The size of the kernels/filters we use for convolution, the number of layers and neurons, the proportion we choose to drop in each layer, when we flatten, our optimizer, and other hyperparameters will be adjusted per the use case and will be optimized via a grid search when the time comes. With a good labeled dataset and the use of a CNN with opencv, real time object classification should be possible.




# SUDOKU SOLVE
This portion of the project is the least applicable to a different AR, so I have not written any new code here. In a different use case, this step would be replaced with something different, whether it’s printing the name of an object on the screen, showing other related objects, etc. For Sudoku however, the augmented reality environment is completed by running a known recursive algorithm for solving Sudoku, and then printing the correct numbers in the correct locations on the grid.



# SUDOKU SOLVER

NOTE 1: This is NOT an original project. The concept for the project has already been created, and the project has already been open-sourced. This is merely an attempt to recreate it in my own repository. Large portions of the code, as well as a pre-trained neural network, were taken directly from repositories who have been listed in the references section, where you will find other resources that heavily inspired the code that I did write myself. I do not claim this code, or this project to be my own. This repository is meant to provide instructions and code to get a project of this type up and running.



NOTE 2: The CNN is still not identifying every digit correctly so for the time being the program is struggling to get a complete grid to solve. 



# PROJECT OVERVIEW
I was tasked to reverse engineer code which takes a video of someone holding up a sudoku puzzle as the input, and prints out the exact same video with the entire grid filled in with the solved puzzle. This is a complex task, but it can be thought of in three manageable steps. 

**1.** From the raw video input, isolate the portion of the frame containing the sudoku grid and transform it so it's upright.
**2.** From the grid image, grab the portion of each nonempty box containing the number, and run this image through a digit classifying CNN to receive the digit in the box in order to fill out a matrix with each number on the grid. 
**3.** Using a known Sudoku algorithm, solve this matrix and print out the numerical solutions into their respective spots on the sudoku grid, undo the grid transformation, and return this image to the user.

Each of these steps are critical to this particular use case of opencv, but in order to modify this code to fit a different object classification use case, step 2 (and, to a lesser degree, step 1) is the most critical to understand. 

# SETUP & INSTRUCTIONS

*This is just one possible method to set up and run this project.*

**Step 1: Download Python:** 
In order to run this project, python will have to be installed on your device. You can quickly do this by going [here](https://www.python.org/downloads/) and clicking on the big yellow button which says "Download Python". Make sure you download the correct version for your operating system, and install the downloaded file.


**Step 2: Download Anaconda:**
Anaconda is a python interface which I recommend using to set up this project. You can find it [here](https://www.anaconda.com/products/individual). This will allow you to quickly install the necessary python libraries for this project, and launch the Jupyter Notebook environment.

**Step 3: Install Libraries via Anaconda Prompt:**
There are a few python libraries which need to be installed in order to run this project. Luckily, this is an easy task with Anaconda. Open up the anaconda prompt, a program which should be on your computer after downloading Anaconda. This will open up a program which looks like command prompt/terminal. First we want to install the python linear algebra toolkit, numpy, which can be accomplished by simply entering the command 

    conda install numpy
    
Next we want to install our opencv (computer vision) library. This can be accomplished by typing the following command into anaconda prompt:

    conda install -c conda-forge opencv
    
If reaching this point has led to confusion, [this](https://medium.com/@pranav.keyboard/installing-opencv-for-python-on-windows-using-anaconda-or-winpython-f24dd5c895eb) is a great article detailing how to get opencv installed via anaconda step-by-step with screenshots
    
Now, in order to utilize the convolutional neural network we will be using for number classification, we need to install the python neural network library Keras, which can be done again by typing into anaconda prompt:

    conda install -c anaconda keras
    
Great! Now we have successfully installed the necessary libraries.

**Step 4: Launch Jupyter Notebooks:**
Open up the Anaconda Navigator program which should have been downloaded with Anaconda. Under the home tab, you will see a list of boxes, each of which contains a different python environment. Click on the "launch" box within the larger box entitled "Jupyter Notebooks", which looks like this: 
![mypic0](images/Jupyter.PNG)

This should open up your browser with a folder directory (for windows this is typically your user folder in your C drive).

**Step 5: Paste in code and models to local folder within Jupyter:**
The simplest way to proceed in this step is to navigate to wherever you would like this project to reside on your computer within the Jupyter browser directory, and create a folder for this project. Then simply download the files from this repository, and place them in the folder you've created. 

**Run code, put grid in front of camera:**
The final step is to open up the main.ipynb file and click run. If you have a functioning webcam, a new window should open up displaying your video feed. Print out one of the sample sudoku grids provided, or one of your own, and hold it up to the camera (relatively upright, with a steady hand) and the solved numbers should be printed out on the grid in the video feed.

# BACKGROUND

# Computer Vision Overview
Computer vision is the field concerning the ability of a computer to process photo or video input that it receives from a visual sensor. This is accomplished by converting an image into an array of tuples of three pixels with values between 0 and 255 (representing the RGB color system), and representing a video as a set of images which continually change to reflect the current frame. This array can then be manipulated for a variety of purposes, including motion detection, object classification, filter application, among many others. The use of Python’s opencv library for computer vision which operates on top of Python’s numpy library for linear algebra makes computer vision tasks much simpler than they otherwise would be. 






# Sudoku Grid Detection

The first step is to find the contour, or region of the frame which contains the sudoku grid. 
    


![mypic1](images/grid_grab.PNG)




In short, I set up a loop to constantly be grabbing new camera frames, thus creating a real time video. Then, I took each individual frame and converted it to a black and white image to make image processing possible. Then I applied Gaussian blurring in order to remove image noise, and adaptive Gaussian thresholding to mark tangible “edges” in the photo. I then found all of the contours, which are essentially just the boundary of closed regions formed from these edges. Because the entire frame was being counted as one big contour, I wrote code to grab the second biggest contour, and then to draw this back on the original frame.



After this, the rotated rectangle of the grid must be mapped to an upright rectangle in order for number classification to occur, so we have to use the warpperspective() function to accomplish this. 

    




# NUMBER DETECTION 

The grid is then split up into its 81 individual components by dividing the width and height by 9, accounting for the borders on the sudoku grid, and then "cleaned up" by cropping based on the number's contour. After this, each image of an individual number is resized and prepared to enter the network for recognition. Each image of a number is “looked” at by the computer, input into a convolutional neural network, and output as a digit between 0-9. The beauty of this process is the fact that this is happening in real time is no obstacle, as each individual frame is looked at separately, and its grid located and numbers classified instantaneously.  Each still image will run images through the neural network and receive numbers. The accuracy of this type of numerical classification is over 99%, and this is owed in large part to the power of the convolutional neural network. 


# SUDOKU SOLVE
The grid of numbers which should match the Sudoku board of the user is now solved by importing and running a known recursive algorithm for Sudoku, and then printing the correct numbers in the correct locations on the grid, so we now have a solved, filled out board.



# LAST STEP
Finally, the numbers of the solved grid are printed back onto the grid image, and then the reverse of the earlier perspective transformation is applied to paste this back onto the original image. This means that the solution will be printed onto your paper grid on the screen in real time.


# VERSIONS
Anaconda: 4.8.3
Python: 3.8.3
OpenCV: 4.3.0
Keras: 2.4.3




# REFERENCES

https://github.com/guille0/songoku
This repository was extremely helpful in the pre-processing department. Portions of this code were directly copied, and some were used as inspiration.

https://github.com/anhminhtran235/real_time_sudoku_solver
This repository was very useful in preprocessing the numbers as well as pasting the solutions back onto the image.
I copied a lot of code directly from this repo, used some for for inspiration, as well as using the neural network and sudoku solver files directly (not my own in any capacity) so all credit goes to this repository's owner for these files and large sections of code. The files 'digitRecognition.h5' and 'solver.py' are both taken directly from this repository.


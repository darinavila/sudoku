# SUDOKU SOLVER

NOTE: This is NOT an original project. The concept for the project has already been created, and the project has already been open-sourced. This is merely an attempt to recreate it in my own repository. Large portions of the code, as well as a pre-trained neural network, were taken directly from repositories who have been listed in the references section, where you will find other resources that heavily inspired the code that I did write myself. I do not claim this code, or this project to be my own. This repository is meant to provide instructions and code to get a project of this type up and running.


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
The final step is to open up the main.ipynb file and click run. If you have a functioning webcam, a new window should open up displaying your video feed. Print out ne of the sample sudoku grids provided, or one of your own, and hold it up to the camera (relatively upright, with a steady hand) and the solved numbers should be printed out on the grid in the video feed.




# REFERENCES



# Stereo Matching using Block Matching (Image Gradients)
# Computes a disparity map from a rectified stereo pair using Block Matching

import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from genericFunctions import *

# Set parameters
dispLevels = 16 #disparity range: 0 to dispLevels-1
windowSize = 5

# Define data cost computation
dataCostComputation = lambda left,right: np.sum(np.absolute(left-right),axis=2) #magnitude

# Start timer
timerVal = time.time()

# Load left and right images in grayscale
leftImg = cv.imread("left.png",cv.IMREAD_GRAYSCALE)
rightImg = cv.imread("right.png",cv.IMREAD_GRAYSCALE)

# Apply a Gaussian filter
leftImg = cv.GaussianBlur(leftImg,(5,5),0.6)
rightImg = cv.GaussianBlur(rightImg,(5,5),0.6)

# Get the size
(rows,cols) = leftImg.shape

# Compute image gradients
leftGrad = np.zeros((rows,cols,2),dtype=np.float64)
leftGrad[:,:,0] = cv.Sobel(leftImg,cv.CV_64F,1,0,ksize=3)
leftGrad[:,:,1] = cv.Sobel(leftImg,cv.CV_64F,0,1,ksize=3)
rightGrad = np.zeros((rows,cols,2),dtype=np.float64)
rightGrad[:,:,0] = cv.Sobel(rightImg,cv.CV_64F,1,0,ksize=3)
rightGrad[:,:,1] = cv.Sobel(rightImg,cv.CV_64F,0,1,ksize=3)

# Compute pixel-based matching cost (data cost)
dataCost = np.zeros((rows,cols,dispLevels),dtype=np.float64)
for d in range(dispLevels):
    rightGradShifted = shiftRight(rightGrad,d,0)
    dataCost[:,:,d] = dataCostComputation(leftGrad,rightGradShifted)

# Aggregate the matching cost
dataCost = cv.boxFilter(dataCost,-1,(windowSize,windowSize),normalize=False)

# Compute the disparity map
dispMap = np.argmin(dataCost,axis=2)

# Normalize the disparity map for display
scaleFactor = 256/dispLevels
dispImg = (dispMap*scaleFactor).astype(np.uint8)

# Show disparity map
plt.imshow(dispImg,cmap="gray")
plt.show(block=False)
plt.pause(0.01)

# Save disparity map
cv.imwrite("disparityBM_Grad.png",dispImg)

# Stop timer and display running time
elapsedTime = time.time()-timerVal
print("Running time: {:.2f} seconds".format(elapsedTime))

plt.show()

# Stereo Matching using Belief Propagation (with Bipartite message update schedule)
# Computes a disparity map from a rectified stereo pair using Belief Propagation

import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from shiftArray import shiftArray

# Set parameters
dispLevels = 16 #disparity range: 0 to dispLevels-1
iterations = 60
lambda_ = 5 #weight of smoothness cost
trunc = 2 #truncation of smoothness cost

# Define data cost computation
dataCostComputation = lambda left,right: np.absolute(left-right) #absolute differences
#dataCostComputation = lambda left,right: (left-right)**2 #square differences

# Define smoothness cost computation
smoothnessCostComputation = lambda differences: lambda_*np.minimum(np.absolute(differences),trunc)

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

# Convert to int32
leftImg = leftImg.astype(np.int32)
rightImg = rightImg.astype(np.int32)

# Compute pixel-based matching cost (data cost)
dataCost = np.zeros((rows,cols,dispLevels),dtype=np.int32)
for d in range(dispLevels):
    #rightImgShifted = shiftArray(rightImg,[0,d])
    rightImgShifted = np.roll(rightImg,d,1) #less accurate, better performances
    dataCost[:,:,d] = dataCostComputation(leftImg,rightImgShifted)

# Compute smoothness cost
d = np.arange(dispLevels)
smoothnessCost = smoothnessCostComputation(d-d[np.newaxis,:].T)
smoothnessCost4d = smoothnessCost[np.newaxis,np.newaxis,:,:].astype(np.int32)

# Initialize messages
msgFromUp = np.zeros((rows,cols,dispLevels),dtype=np.int32)
msgFromDown = np.zeros((rows,cols,dispLevels),dtype=np.int32)
msgFromRight = np.zeros((rows,cols,dispLevels),dtype=np.int32)
msgFromLeft = np.zeros((rows,cols,dispLevels),dtype=np.int32)

energy = np.zeros(iterations,dtype=np.int32)
mask = ((np.arange(rows)[:,np.newaxis]+np.arange(cols)[np.newaxis,:])%2)[:,:,np.newaxis]

# Start iterations
for it in range(iterations):
    for i in range(2):
        # Create messages to up
        msgFromDown2 = dataCost + msgFromDown + msgFromRight + msgFromLeft
        msgFromDown2 = np.amin(msgFromDown2[:,:,:,np.newaxis]+smoothnessCost4d,axis=2)
        msgFromDown2 = shiftArray(msgFromDown2,[-1,0,0]) #shift up
        
        # Create messages to down
        msgFromUp2 = dataCost + msgFromUp + msgFromRight + msgFromLeft
        msgFromUp2 = np.amin(msgFromUp2[:,:,:,np.newaxis]+smoothnessCost4d,axis=2)
        msgFromUp2 = shiftArray(msgFromUp2,[1,0,0]) #shift down
        
        # Create messages to right
        msgFromLeft2 = dataCost + msgFromUp + msgFromDown + msgFromLeft
        msgFromLeft2 = np.amin(msgFromLeft2[:,:,:,np.newaxis]+smoothnessCost4d,axis=2)
        msgFromLeft2 = shiftArray(msgFromLeft2,[0,1,0]) #shift right

        # Create messages to left
        msgFromRight2 = dataCost + msgFromUp + msgFromDown + msgFromRight
        msgFromRight2 = np.amin(msgFromRight2[:,:,:,np.newaxis]+smoothnessCost4d,axis=2)
        msgFromRight2 = shiftArray(msgFromRight2,[0,-1,0]) #shift left
        
        # Send messages
        mask1 = (mask!=i); mask2 = (mask==i)
        msgFromDown = msgFromDown*mask1 + msgFromDown2*mask2
        msgFromUp = msgFromUp*mask1 + msgFromUp2*mask2
        msgFromLeft = msgFromLeft*mask1 + msgFromLeft2*mask2
        msgFromRight = msgFromRight*mask1 + msgFromRight2*mask2

    # Normalize messages
    msgFromDown = msgFromDown-np.amin(msgFromDown,axis=2)[:,:,np.newaxis]
    msgFromUp = msgFromUp-np.amin(msgFromUp,axis=2)[:,:,np.newaxis]
    msgFromLeft = msgFromLeft-np.amin(msgFromLeft,axis=2)[:,:,np.newaxis]
    msgFromRight = msgFromRight-np.amin(msgFromRight,axis=2)[:,:,np.newaxis]

    # Compute belief
    #belief = dataCost + msgFromUp + msgFromDown + msgFromRight + msgFromLeft #standard belief computation
    belief = msgFromUp + msgFromDown + msgFromRight + msgFromLeft #without dataCost (larger energy but better results)
    
    # Compute the disparity map
    dispMap = np.argmin(belief,axis=2)
    
    # Compute energy
    dataEnergy = np.sum(dataCost[np.arange(rows)[:,np.newaxis],np.arange(cols)[np.newaxis,:],dispMap])
    smoothnessEnergyHorizontal = np.sum(smoothnessCostComputation(np.diff(dispMap,n=1,axis=1)))
    smoothnessEnergyVertical = np.sum(smoothnessCostComputation(np.diff(dispMap,n=1,axis=0)))
    energy[it] = dataEnergy+smoothnessEnergyHorizontal+smoothnessEnergyVertical

    # Normalize the disparity map for display
    scaleFactor = 256/dispLevels
    dispImg = (dispMap*scaleFactor).astype(np.uint8)

    # Show disparity map
    plt.cla()
    plt.imshow(dispImg,cmap="gray")
    plt.show(block=False)
    plt.pause(0.01)

    # Show energy and iteration
    print("iteration: {0}/{1}, energy: {2}".format(it+1,iterations,energy[it]))

# Show convergence graph
plt.figure()
plt.plot(np.arange(1,iterations+1),energy,marker="o")
plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.show(block=False)
plt.pause(0.01)

# Save disparity map
cv.imwrite("disparityBP_Bipart.png",dispImg)

# Stop timer and display running time
elapsedTime = time.time()-timerVal
print("Running time: {:.2f} seconds".format(elapsedTime))

plt.show()

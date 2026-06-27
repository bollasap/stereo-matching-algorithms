import numpy as np

def shiftDown(A,n,fillValue):
    """shiftDown Shift an array DOWN with specific padding."""
    B = np.roll(A,n,0)
    B[:n] = fillValue
    return B

def shiftUp(A,n,fillValue):
    """shiftUp Shift an array UP with specific padding."""
    B = np.roll(A,-n,0)
    B[-n:] = fillValue
    return B

def shiftRight(A,n,fillValue):
    """shiftRight Shift an array RIGHT with specific padding."""
    B = np.roll(A,n,1)
    B[:,:n] = fillValue
    return B

def shiftLeft(A,n,fillValue):
    """shiftLeft Shift an array LEFT with specific padding."""
    B = np.roll(A,-n,1)
    B[:,-n:] = fillValue
    return B

def shiftForward(A,n,fillValue):
    """shiftForward Shift an array FORWARD with specific padding."""
    B = np.roll(A,n,2)
    B[:,:,:n] = fillValue
    return B

def shiftBackward(A,n,fillValue):
    """shiftBackward Shift an array BACKWARD with specific padding."""
    B = np.roll(A,-n,2)
    B[:,:,-n:] = fillValue
    return B

def shiftYX(A,y,x,fillValue):
    """shiftYX Shift an array by dimensions Y and X with specific padding."""
    B = np.roll(A,(y,x),(0,1))
    if (y > 0):
        B[:y] = fillValue
    elif (y < 0):
        B[y:] = fillValue
    if (x > 0):
        B[:,:x] = fillValue
    elif (x < 0):
        B[:,x:] = fillValue
    return B

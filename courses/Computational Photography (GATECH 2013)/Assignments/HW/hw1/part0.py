import sys
import os
import numpy as np
import random
import cv2
import run

def convolve(image, kernel):
  '''Convolve the given image and kernel

  Inputs:

  image - a single channel (rows, cols) image with dtype uint8

  kernel - a matrix of shape (d, d) and dtype float

  Outputs:

  output - an output array of the same dtype as image, and of shape
  (rows - d + 1, cols - d + 1)
 
  Every element of output should contain the application of the kernel to the
  corresponding location in the image.

  Output elements that result in values that are greater than 255 should be 
  cropped to 255, and output values less than 0 should be set to 0.
  '''
  output = None
  # Insert your code here.----------------------------------------------------

  #---------------------------------------------------------------------------
  return output 

def test():
  '''This script will perform a unit test on your function, and provide useful
  output.
  '''
  images = []
  kernels = []
  outputs = []

  x = np.zeros((9,9), dtype = np.uint8)
  x[4,4] = 255 
  images.append(x)

  x = np.zeros((5,5), dtype = np.uint8)
  x[2,2] = 255 
  images.append(x)

  y = np.array(
      [[-0.9,-0.7, 0.5, 0.3, 0.0],
       [-0.7, 0.5, 0.3, 0.0, 0.4],
       [ 0.5, 0.1, 0.0, 0.4, 0.6],
       [ 0.1, 0.0, 0.2, 0.6, 0.8],
       [ 0.0, 0.2, 0.6, 0.8, 1.2]])
  kernels.append(y)

  y = np.array(
      [[0.1, 0.1, 0.1],
       [0.1, 0.2, 0.1],
       [0.1, 0.1, 0.1]])
  kernels.append(y)

  z = np.array(
      [[   0,   0, 127,  76,   0],
       [   0, 127,  76,   0, 102],
       [ 127,  25,   0, 102, 153],
       [  25,   0,  51, 153, 204],
       [   0,  51, 153, 204, 255]], dtype = np.uint8)
  outputs.append(z)

  z = np.array(
      [[ 25,  25,  25],
       [ 25,  51,  25],
       [ 25,  25,  25]], dtype = np.uint8)
  outputs.append(z)

  for image, kernel, output in zip(images, kernels, outputs):
    if __name__ == "__main__":
      print "image:\n{}".format(image)
      print "kernel:\n{}".format(kernel)

    usr_out = convolve(image, kernel)

    if not type(usr_out) == type(output):
      if __name__ == "__main__":
        print "Error- output has type {}. Expected type is {}.".format(
            type(usr_out), type(output))
      return False

    if not usr_out.shape == output.shape:
      if __name__ == "__main__":
        print "Error- output has shape {}. Expected shape is {}.".format(
            usr_out.shape, output.shape)
      return False

    if not usr_out.dtype == output.dtype:
      if __name__ == "__main__":
        print "Error- output has dtype {}. Expected dtype is {}.".format(
            usr_out.dtype, output.dtype)
      return False

    if not np.all(usr_out == output):
      if __name__ == "__main__":
        print "Error- output has value:\n{}\nExpected value:\n{}".format(
            usr_out, output)
      return False
    if __name__ == "__main__":
      print "Passed."

  if __name__ == "__main__":
    print "Success."
  return True

if __name__ == "__main__":
  # Testing code
  print "Performing unit test."
  test()

import sys
import os
import numpy as np
from scipy.stats import norm
import math
import random
import cv2
import run

def filter_median(image, k):
  '''Filter the image using a median kernel.
  
  Inputs:

  image - a single channel image of shape (rows, cols)

  k - the radius of the neighborhood you should use (positive integer)

  Output:

  output - a numpy array of shape (rows - 2k, cols - 2k) and the same dtype as 
  image.
  
  Each cell in the output image should be filled with the median value of the
  corresponding (2k+1, 2k+1) patch in the image.
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
  x = np.array([[   0,   1,   2,   3,   4],
                [   5,   6,   7,   8,   9],
                [  10,  11,  12,  13,  14],
                [  15,  16,  17,  18,  19],
                [  20,  21,  22,  23,  24]], dtype = np.uint8)
  images.append(x)
  images.append(x)

  x = np.array([[ 0,  1,  2,  3,  4,  5,  6],
                [ 7,  8,  9, 10, 11, 12, 13],
                [14, 15, 16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25, 26, 27],
                [28, 29, 30, 31, 32, 33, 34],
                [35, 36, 37, 38, 39, 40, 41],
                [42, 43, 44, 45, 46, 47, 48]], dtype = np.uint8)
  images.append(x)
  images.append(x)

  ks = [1, 2, 1, 2]
  outputs = []

  z = np.array([[ 6,  7,  8],
                [11, 12, 13],
                [16, 17, 18]], dtype=np.uint8)
  outputs.append(z)

  z = np.array([[12]], dtype=np.uint8)
  outputs.append(z)

  z = np.array([[ 8,  9, 10, 11, 12],
                [15, 16, 17, 18, 19],
                [22, 23, 24, 25, 26],
                [29, 30, 31, 32, 33],
                [36, 37, 38, 39, 40]], dtype=np.uint8)
  outputs.append(z)

  z = np.array([[16, 17, 18],
                [23, 24, 25],
                [30, 31, 32]], dtype=np.uint8)
  outputs.append(z)

  for image, k, output in zip(images, ks, outputs):
    if __name__ == "__main__":
      print "image:\n{}".format(image)
      print "k:\n{}".format(k)

    usr_out = filter_median(image, k)

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

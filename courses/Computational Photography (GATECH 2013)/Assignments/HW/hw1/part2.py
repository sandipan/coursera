import sys
import os
import numpy as np
from scipy.stats import norm
import math
import random
import cv2
import run

def make_sharp(k, sd):
  '''Create a sharpen kernel.
  
  Input:

  k - the radius of the kernel.
  sd - the standard deviation of the gaussian filter used to make the kernel.
  
  Output:

  output - a numpy array of shape (2k+1, 2k+1) and dtype float.
  
  The sharpen filter is constructed by first taking a filter with a 2 in the
  center and 0's everywhere else, and subtracting from that a gaussian filter.
  
  Note:

  You can use the make_gaussian function from part one by typing:

  import part1
  part1.make_gaussian(k, sd)
  '''
  kernel = None
  # Insert your code here.----------------------------------------------------

  #---------------------------------------------------------------------------
  return kernel

def test():
  '''This script will perform a unit test on your function, and provide useful
  output.
  '''

  np.set_printoptions(precision=3)

  ks =  [1, 2, 1, 2, 1]
  sds = [1, 2, 3, 4, 5]
  outputs = []
  # 1,1
  y = np.array([[-0.075, -0.124, -0.075],
                [-0.124,  1.796, -0.124],
                [-0.075, -0.124, -0.075]])
  outputs.append(y)
  # 2,2
  y = np.array([[-0.023, -0.034, -0.038, -0.034, -0.023],
                [-0.034, -0.049, -0.056, -0.049, -0.034],
                [-0.038, -0.056,  1.937, -0.056, -0.038],
                [-0.034, -0.049, -0.056, -0.049, -0.034],
                [-0.023, -0.034, -0.038, -0.034, -0.023]])
  outputs.append(y)
  # 1,3
  y = np.array([[-0.107, -0.113, -0.107],
                [-0.113,  1.880, -0.113],
                [-0.107, -0.113, -0.107]])
  outputs.append(y)
  # 2,4
  y = np.array([[-0.035, -0.039, -0.04 , -0.039, -0.035],
                [-0.039, -0.042, -0.044, -0.042, -0.039],
                [-0.04 , -0.044,  1.955, -0.044, -0.04 ],
                [-0.039, -0.042, -0.044, -0.042, -0.039],
                [-0.035, -0.039, -0.04 , -0.039, -0.035]])
  outputs.append(y)
  # 1,5
  y = np.array([[-0.11 , -0.112, -0.11 ],
                [-0.112,  1.886, -0.112],
                [-0.11 , -0.112, -0.11 ]])
  outputs.append(y)

  for k, sd, output in zip(ks, sds, outputs):
    if __name__ == "__main__":
      print "k:{}, sd:{}".format(k, sd)

    usr_out = make_sharp(k, sd)

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

    if not np.all(np.abs(usr_out - output) < .005):
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
  print "Performing unit test. Answers will be accepted as long as they are \
within .005 of the input."
  test()

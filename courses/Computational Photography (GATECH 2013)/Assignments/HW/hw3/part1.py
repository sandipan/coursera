import numpy as np
import scipy.signal
import cv2
import math

import part0

def gauss_pyramid(image, levels):
  '''Construct a pyramid from the image, of height levels.

  image - an image of dimension (r,c) and dtype float.
  levels - a positive integer that specifies the number of reductions you should 
           do. So, if levels = 0, you should return a list containing just the 
           input image. If levels = 1, you should do one reduction, etc. 
           len(output) = levels + 1
  output - a list of arrays of dtype np.float. The first element of the list is
           layer 0 of the pyramid (the image itself). output[1] is layer 1 of the
           pyramid (image reduced once), etc.

  Consult the lecture and tutorial videos for more details about Gaussian Pyramids.
  '''
  output = []
  # Insert your code here ------------------------------------------------------

  # ----------------------------------------------------------------------------
  return output

def lapl_pyramid(gauss_pyr):
  '''Construct a laplacian pyramid from the gaussian pyramid, of height levels.

  gauss_pyr - a gaussian pyramid as returned by your gauss_pyramid function.

  output - a laplacian pyramid of the same height as gauss_pyr. This pyramid
           should be represented in the same way as guass_pyr - as a list of
           arrays. Every element of the list now corresponds to a layer of the 
           laplacian pyramid, containing the difference between two layers of 
           the gaussian pyramid. 

           output[k] = gauss_pyr[k] - expand(gauss_pyr[k+1])

           The last element of output should be identical to the last layer of
           the input pyramid.

  Note: sometimes the size of the expanded image will be larger than the given
  layer. You should crop the expanded image to match in shape with the given
  layer.

  For example, if my layer is of size 5x7, reducing and expanding will result
  in an image of size 6x8. In this case, crop the expanded layer to 5x7.
  '''
  output = []
  # Insert your code here ------------------------------------------------------

  # ----------------------------------------------------------------------------
  return output

def test():
  '''This script will perform a unit test on your function, and provide useful
  output.
  '''
  gauss_pyr1 =[np.array([[   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                         [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                         [   0.,    0.,  255.,  255.,  255.,  255.,    0.,    0.],
                         [   0.,    0.,  255.,  255.,  255.,  255.,    0.,    0.],
                         [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                         [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.]]),
               np.array([[   0.64,    8.92,   12.11,    3.82],
                         [   8.29,  116.03,  157.46,   49.73],
                         [   3.82,   53.55,   72.67,   22.95]]),
               np.array([[ 12.21,  31.85],
                         [ 17.62,  45.97]]),
               np.array([[ 9.77]])] 

  gauss_pyr2 = [np.array([[ 255.,  255.,  255.,  255.,  255.,  255.,  255.],
                          [ 255.,  255.,  255.,  255.,  255.,  255.,  255.],
                          [ 255.,  255.,  125.,  125.,  125.,  255.,  255.],
                          [ 255.,  255.,  125.,  125.,  125.,  255.,  255.],
                          [   0.,    0.,    0.,    0.,    0.,    0.,    0.]]),
                np.array([[ 124.62,  173.95,  173.95,  124.62],
                          [ 165.35,  183.1 ,  183.1 ,  165.35],
                          [  51.6 ,   49.2 ,   49.2 ,   51.6 ]]),
                np.array([[  72.85,  104.71],
                          [  49.53,   68.66]]),
                np.array([[ 31.37]])] 

  if __name__ == "__main__":
    print 'Evaluating gauss_pyramid.'

  for pyr in gauss_pyr1, gauss_pyr2:
    if __name__ == "__main__":
      print "input:\n{}\n".format(pyr[0])

    usr_out = gauss_pyramid(pyr[0], 3)

    if not type(usr_out) == type(pyr):
      if __name__ == "__main__":
        print "Error- gauss_pyramid out has type {}. Expected type is {}.".format(
            type(usr_out), type(pyr))
      return False

    if not len(usr_out) == len(pyr):
      if __name__ == "__main__":
        print "Error- gauss_pyramid out has len {}. Expected len is {}.".format(
            len(usr_out), len(pyr))
      return False

    for usr_layer, true_layer in zip(usr_out, pyr):
      if not type(usr_layer) == type(true_layer):
        if __name__ == "__main__":
          print "Error- output layer has type {}. Expected type is {}.".format(
              type(usr_layer), type(true_layer))
        return False

      if not usr_layer.shape == true_layer.shape:
        if __name__ == "__main__":
          print "Error- gauss_pyramid layer has shape {}. Expected shape is {}.".format(
              usr_layer.shape, true_layer.shape)
        return False

      if not usr_layer.dtype == true_layer.dtype:
        if __name__ == "__main__":
          print "Error- gauss_pyramid layer has dtype {}. Expected dtype is {}.".format(
              usr_layer.dtype, true_layer.dtype)
        return False

      if not np.all(np.abs(usr_layer - true_layer) < 1):
        if __name__ == "__main__":
          print "Error- gauss_pyramid layer has value:\n{}\nExpected value:\n{}".format(
              usr_layer, true_layer)
        return False

  if __name__ == "__main__":
    print "gauss_pyramid passed.\n"
    print "Evaluating lapl_pyramid."

  lapl_pyr1 =[np.array([[  -2.95,  -10.04,  -17.67,  -22.09,  -23.02,  -16.73,   -8.97,   -4.01],
                        [  -9.82,  -33.47,  -58.9 ,  -73.63,  -76.75,  -55.78,  -29.9 ,  -13.39],
                        [ -15.57,  -53.07,  161.59,  138.24,  133.29,  166.55,  -47.41,  -21.23],
                        [ -13.32,  -45.42,  175.06,  155.07,  150.83,  179.3 ,  -40.58,  -18.17],
                        [  -8.55,  -29.16,  -51.33,  -64.16,  -66.88,  -48.61,  -26.05,  -11.67],
                        [  -4.21,  -14.34,  -25.24,  -31.55,  -32.89,  -23.91,  -12.81,   -5.74]]),
              np.array([[ -11.59,  -11.88,  -13.1 ,  -11.22],
                        [  -7.53,   89.12,  124.84,   30.27],
                        [ -12.43,   25.91,   39.17,    2.97]]),
              np.array([[  5.96,  27.94],
                        [ 13.71,  43.53]]),
              np.array([[ 9.77]])] 

  lapl_pyr2 =[np.array([[ 146.27,  118.15,  101.65,   97.53,  101.65,  118.15,  146.27],
                        [ 121.16,   93.25,   79.83,   76.48,   79.83,   93.25,  121.16],
                        [ 118.2 ,   95.65,  -41.91,  -43.79,  -41.91,   95.65,  118.2 ],
                        [ 156.61,  142.69,    9.62,    8.85,    9.62,  142.69,  156.6 ],
                        [ -52.02,  -57.74,  -57.68,  -57.67,  -57.68,  -57.74,  -52.02]]),
              np.array([[  64.97,   97.02,   95.12,   79.3 ],
                        [ 107.73,  109.16,  107.63,  122.01],
                        [   7.53,   -6.95,   -7.81,   18.9 ]]),
              np.array([[ 52.77,  92.16],
                        [ 36.98,  60.82]]),
              np.array([[ 31.37]])] 

  for gauss_pyr, lapl_pyr in zip((gauss_pyr1, gauss_pyr2), (lapl_pyr1, lapl_pyr2)):
    if __name__ == "__main__":
      print "input:\n{}".format(gauss_pyr)

    usr_out = lapl_pyramid(gauss_pyr)

    if not type(usr_out) == type(lapl_pyr):
      if __name__ == "__main__":
        print "Error- lapl_pyramid out has type {}. Expected type is {}.".format(
            type(usr_out), type(lapl_pyr))
      return False

    if not len(usr_out) == len(lapl_pyr):
      if __name__ == "__main__":
        print "Error- lapl_pyramid out has len {}. Expected len is {}.".format(
            len(usr_out), len(lapl_pyr))
      return False

    for usr_layer, true_layer in zip(usr_out, lapl_pyr):
      if not type(usr_layer) == type(true_layer):
        if __name__ == "__main__":
          print "Error- output layer has type {}. Expected type is {}.".format(
              type(usr_layer), type(true_layer))
        return False

      if not usr_layer.shape == true_layer.shape:
        if __name__ == "__main__":
          print "Error- lapl_pyramid layer has shape {}. Expected shape is {}.".format(
              usr_layer.shape, true_layer.shape)
        return False

      if not usr_layer.dtype == true_layer.dtype:
        if __name__ == "__main__":
          print "Error- lapl_pyramid layer has dtype {}. Expected dtype is {}.".format(
              usr_layer.dtype, true_layer.dtype)
        return False

      if not np.all(np.abs(usr_layer - true_layer) < 1):
        if __name__ == "__main__":
          print "Error- lapl_pyramid layer has value:\n{}\nExpected value:\n{}".format(
              usr_layer, true_layer)
        return False

  if __name__ == "__main__":
    print "lapl_pyramid passed."

  if __name__ == "__main__":
    print "All unit tests successful."
  return True

if __name__ == "__main__":
  print "Performing unit tests. Your functions will be accepted if your result is\
    within 1 of the correct output."
  np.set_printoptions(precision=1)

  test()

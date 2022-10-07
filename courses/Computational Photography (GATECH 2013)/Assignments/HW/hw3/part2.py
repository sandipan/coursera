import numpy as np
import scipy.signal
import cv2
import math

import part0

def blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):
  '''Blend the two laplacian pyramids by weighting them according to
  the gaussian mask.

  lapl_pyr_white - a laplacian pyramid of one image, as constructed by your
                   lapl_pyramid function.

  lapl_pyr_black - a laplacian pyramid of another image, as constructed by your
                   lapl_pyramid function.

  gauss_pyr_mask - a gaussian pyramid of the mask. Each value is in the range
                   [0, 1].

  The pyramids will have the same number of levels. Furthermore, each layer is
  guaranteed to have the same shape.

  You should return a laplacian pyramid that is of the same dimensions as the 
  input pyramids. Every layer should be an alpha-blend of the corresponding layers
  of the input pyramids, weighted by the gaussian mask.

  Pixels where gauss_pyr_mask == 1 should be taken completely from the white image.
  Pixels where gauss_pyr_mask == 0 should be taken completely from the black image.
  '''

  blended_pyr = []
  # Insert your code here ------------------------------------------------------

  # ----------------------------------------------------------------------------
  return blended_pyr

def collapse(lapl_pyr):
  '''Reconstruct the image based on its laplacian pyramid.

  lapl_pyr - a laplacian pyramid, as constructed by the lapl_pyramid function, or
             returned by the blend function.

  output - an image of the same shape as the base layer of the pyramid and dtype
           float.

  Note: sometimes expand will return an image that is larger than the next layer.
  In this case, you should crop the expanded image down to the size of the next
  layer.

  For example, expanding a layer of size 3x4 will result in an image of size 6x8.
  If the next layer is of size 5x7, crop the expanded image to size 5x7.
  '''
  output = None
  # Insert your code here ------------------------------------------------------

  # ----------------------------------------------------------------------------
  return output

def test():
  '''This script will perform a unit test on your function, and provide useful
  output.
  '''
  lapl_pyr11 =[np.array([[ 0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.]]),
                np.array([[ 0.,  0.],
                          [ 0.,  0.]])] 
  lapl_pyr12 =[np.array([[ 149.77,  122.46,  121.66,  178.69],
                          [ 138.08,  107.74,  106.84,  170.21],
                          [ 149.77,  122.46,  121.66,  178.69]]),
                np.array([[ 124.95,  169.58],
                          [ 124.95,  169.57]])] 
  lapl_pyr21 =[np.array([[ 149. ,  118.4,   99.2,   94.3,   99.2,  118.4,  149. ],
                          [ 137.2,  103.3,   81.9,   76.5,   81.9,  103.3,  137.2],
                          [ 148.1,  117.4,   97.9,   93.1,   97.9,  117.4,  148.1],
                          [ -63.1,  -81.3,  -92.8,  -95.6,  -92.8,  -81.3,  -63.1],
                          [ -18.5,  -23.8,  -27.2,  -28. ,  -27.2,  -23.8,  -18.5]]),
                np.array([[  70.4,  107.1,  104.5,   82.3],
                          [  76.7,  115.4,  113.1,   87.3],
                          [ -23.3,  -29.4,  -31. ,  -16.3]]),
                np.array([[  67.7,  100.3],
                          [  34. ,   50.4]])] 
  lapl_pyr22 =[np.array([[  -5. ,  -25.2,  -56.4,  149.8,  110.3,  116.2,  144.8],
                          [  -6.5,  -32.5,  -72.6,  119.5,   68.6,   76.2,  113. ],
                          [  -7.2,  -36. ,  -80.3,  105.2,   48.9,   57.2,   98. ],
                          [  -6.5,  -32.5,  -72.6,  119.5,   68.6,   76.2,  113. ],
                          [  -5. ,  -25.2,  -56.4,  149.8,  110.3,  116.2,  144.8]]),
                np.array([[ -20.9,    4.8,  102.6,   84.1],
                          [ -23.2,   22.3,  167.9,  133.1],
                          [ -20.9,    4.8,  102.6,   84.1]]),
                np.array([[ 17.6,  90.8],
                          [ 17.6,  90.8]])] 
  mask_pyr1 =[np.array([[ 0.,  0.,  1.,  1.],
                        [ 0.,  0.,  1.,  1.],
                        [ 0.,  0.,  1.,  1.]]),
              np.array([[ 0.03,  0.46],
                        [ 0.03,  0.46]])] 
  mask_pyr2 = [np.array([[ 0.,  0.,  0.,  0.,  1.,  1.,  1.],
                         [ 0.,  0.,  0.,  0.,  1.,  1.,  1.],
                         [ 0.,  0.,  0.,  0.,  1.,  1.,  1.],
                         [ 0.,  0.,  0.,  0.,  1.,  1.,  1.],
                         [ 0.,  0.,  0.,  0.,  1.,  1.,  1.]]),
               np.array([[ 0. ,  0. ,  0.5,  0.5],
                         [ 0. ,  0. ,  0.7,  0.7],
                         [ 0. ,  0. ,  0.5,  0.5]]),
               np.array([[ 0. ,  0.3],
                         [ 0. ,  0.3]])] 
  out_pyr1 =[np.array([[ 149.77,  122.46,    0.  ,    0.  ],
                       [ 138.08,  107.74,    0.  ,    0.  ],
                       [ 149.77,  122.46,    0.  ,    0.  ]]),
             np.array([[ 120.58,   92.42],
                       [ 120.58,   92.42]])] 
  out_pyr2 = [np.array([[  -5. ,  -25.2,  -56.4,  149.8,   99.2,  118.4,  149. ],
                        [  -6.5,  -32.5,  -72.6,  119.5,   81.9,  103.3,  137.2],
                        [  -7.2,  -36. ,  -80.3,  105.2,   97.9,  117.4,  148.1],
                        [  -6.5,  -32.5,  -72.6,  119.5,  -92.8,  -81.3,  -63.1],
                        [  -5. ,  -25.2,  -56.4,  149.8,  -27.2,  -23.8,  -18.5]]),
              np.array([[ -20.9,    4.8,  103.5,   83.2],
                        [ -23.2,   22.3,  129.5,  101. ],
                        [ -20.9,    4.8,   35.8,   33.9]]),
              np.array([[ 17.6,  93.6],
                        [ 17.6,  78.7]])] 
  outimg1 = np.array([[ 244.91,  218.31,   77.39,   41.59],
                      [ 243.79,  214.24,   85.99,   46.21],
                      [ 244.91,  218.31,   77.39,   41.59]]) 
  outimg2 = np.array([[   0.1,    0.1,   -0.1,  253.7,  241.3,  254. ,  256. ],
                      [  -0.3,   -0.5,   -2.7,  244.4,  250.3,  263.3,  263.2],
                      [  -0.6,   -1.4,   -6. ,  233.4,  267.8,  278.2,  274.6],
                      [  -0.9,   -2.1,   -8.7,  224.1,   42.2,   46.1,   37.3],
                      [  -1. ,   -2.4,   -9.6,  221.2,   61.5,   59.5,   47.5]])

  if __name__ == "__main__":
    print 'Evaluating blend.'

  for left_pyr, right_pyr, mask_pyr, out_pyr in ((lapl_pyr11, lapl_pyr12, mask_pyr1, out_pyr1), 
      (lapl_pyr21, lapl_pyr22, mask_pyr2, out_pyr2)):
    usr_out = blend(left_pyr, right_pyr, mask_pyr)

    if not type(usr_out) == type(out_pyr):
      if __name__ == "__main__":
        print "Error- output layer has type {}. Expected type is {}.".format(
            type(usr_out), type(out_pyr))
      return False

    if not len(usr_out) == len(out_pyr):
      if __name__ == "__main__":
        print "Error- blend out has len {}. Expected len is {}.".format(
            len(usr_out), len(out_pyr))
      return False

    for usr_layer, true_layer, left_layer, right_layer, mask_layer in zip(usr_out, out_pyr, 
        left_pyr, right_pyr, mask_pyr):
      if not type(usr_layer) == type(true_layer):
        if __name__ == "__main__":
          print "Error- blend out has type {}. Expected type is {}.".format(
              type(usr_layer), type(true_layer))
        return False

      if not usr_layer.shape == true_layer.shape:
        if __name__ == "__main__":
          print "Error- blend output layer has shape {}. Expected shape is {}.".format(
              usr_layer.shape, true_layer.shape)
        return False

      if not usr_layer.dtype == true_layer.dtype:
        if __name__ == "__main__":
          print "Error- blend output layer has dtype {}. Expected dtype is {}.".format(
              usr_layer.dtype, true_layer.dtype)
        return False

      if not np.all(np.abs(usr_layer - true_layer) < 1):
        if __name__ == "__main__":
          print "Error- blend output layer has value:\n{}\nExpected value:\n{}\nInput left:\n{}\nInput right:\n{}\nInput mask:\n{}".format(
              usr_layer, true_layer, left_layer, right_layer, mask_layer)
        return False

  if __name__ == "__main__":
    print "blend passed.\n"
    print "Evaluating collapse."

  for pyr, img in ((out_pyr1, outimg1),(out_pyr2, outimg2)):
    if __name__ == "__main__":
      print "input:\n{}".format(pyr)

    usr_out = collapse(pyr)

    if not type(usr_out) == type(img):
      if __name__ == "__main__":
        print "Error- collapse out has type {}. Expected type is {}.".format(
            type(usr_out), type(img))
      return False

    if not usr_out.shape == img.shape:
      if __name__ == "__main__":
        print "Error- collapse out has shape {}. Expected shape is {}.".format(
            usr_out.shape, img.shape)
      return False

    if not usr_out.dtype == img.dtype:
      if __name__ == "__main__":
        print "Error- collapse out has dtype {}. Expected dtype is {}.".format(
            usr_out.dtype, img.dtype)
      return False

    if not np.all(np.abs(usr_out - img) < 1):
      if __name__ == "__main__":
        print "Error- collapse out has value:\n{}\nExpected value:\n{}".format(
            usr_out, img)
      return False

  if __name__ == "__main__":
    print "collapse passed."

  if __name__ == "__main__":
    print "All unit tests successful."
  return True

if __name__ == "__main__":
  print "Performing unit tests. Your functions will be accepted if your result is\
    within 2 of the correct output."
  np.set_printoptions(precision=1, suppress=True)

  test()

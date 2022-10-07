import numpy as np
import scipy.signal
import cv2

def generating_kernel(a):
  '''Return a 5x5 generating kernel with parameter a.
  '''
  w_1d = np.array([0.25 - a/2.0, 0.25, a, 0.25, 0.25 - a/2.0])
  return np.outer(w_1d, w_1d)

def reduce(image):
  '''Reduce the image to half the size.

  image - a float image of shape (r, c)
  output - a float image of shape (ceil(r/2), ceil(c/2))
  
  For instance, if the input is 5x7, the output will be 3x4.

  You should filter the image with a generating kernel of a = 0.4, and then
  sample every other point.

  Please consult the lectures and tutorial videos for a more in-depth discussion
  of the reduce function.
  '''
  out = None
  # Insert your code here ------------------------------------------------------

  # ----------------------------------------------------------------------------
  return out
  
def expand(image):
  '''Expand the image to double the size.

  image - a float image of shape (r, c)
  output - a float image of shape (2*r, 2*c)

  You should upsample the image, and then filter it with a generating kernel of
  a = 0.4. Finally, scale the output by the appropraite amount to make sure that
  the net weight contributing to each output pixel is 1.

  Please consult the lectures and tutorial videos for a more in-depth discussion
  of the expand function.
  '''
  out = None
  # Insert your code here ------------------------------------------------------

  # ----------------------------------------------------------------------------
  return out

def test():
  '''This script will perform a unit test on your function, and provide useful
  output.
  '''
  # Each subsequent layer is a reduction of the previous one
  reduce1 =[np.array([[   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
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

  reduce2 = [np.array([[ 255.,  255.,  255.,  255.,  255.,  255.,  255.],
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
    print 'Evaluating reduce.'
  for red_pyr in reduce1, reduce2:
    for imgin, true_out in zip(red_pyr[0:-1], red_pyr[1:]):
      if __name__ == "__main__":
        print "input:\n{}\n".format(imgin)

      usr_out = reduce(imgin)

      if not type(usr_out) == type(true_out):
        if __name__ == "__main__":
          print "Error- reduce out has type {}. Expected type is {}.".format(
              type(usr_out), type(true_out))
        return False

      if not usr_out.shape == true_out.shape:
        if __name__ == "__main__":
          print "Error- reduce out has shape {}. Expected shape is {}.".format(
              usr_out.shape, true_out.shape)
        return False

      if not usr_out.dtype == true_out.dtype:
        if __name__ == "__main__":
          print "Error- reduce out has dtype {}. Expected dtype is {}.".format(
              usr_out.dtype, true_out.dtype)
        return False

      if not np.all(np.abs(usr_out - true_out) < 1):
        if __name__ == "__main__":
          print "Error- reduce out has value:\n{}\nExpected value:\n{}".format(
              usr_out, true_out)
        return False

  if __name__ == "__main__":
    print "reduce passed.\n"
    print "Evaluating expand."

  expandin = [np.array([[255]]),
              np.array([[125, 255],
                        [255,   0]]),
              np.array([[ 255.,    0.,  125.,  125.,  125.],
                        [ 255.,    0.,  125.,  125.,  125.],
                        [  50.,   50.,   50.,   50.,   50.]])] 

  expandout =[np.array([[ 163.2 ,  102.  ],
                        [ 102.  ,   63.75]]),
              np.array([[ 120.8 ,  164.75,  175.75,  102.  ],
                        [ 164.75,  158.75,  121.  ,   63.75],
                        [ 175.75,  121.  ,   42.05,   12.75],
                        [ 102.  ,   63.75,   12.75,    0.  ]]),
              np.array([[ 183.6, 114.75, 34.2, 56.25, 101.25, 112.5, 112.5,112.5, 101.25,  56.25],
                        [ 204. ,  127.5,  38.,  62.5,  112.5,  125.,  125., 125.,  112.5,  62.5 ],
                        [ 188.1, 119.75, 39.2, 61.25, 106.25, 117.5, 117.5,117.5, 105.75,  58.75],
                        [ 124.5,  88.75, 44. , 56.25,  81.25,  87.5,  87.5, 87.5,  78.75,  43.75],
                        [  56.4,  52.75, 43.8, 46.25,  51.25,  52.5,  52.5, 52.5,  47.25,  26.25],
                        [  22.5,    25.,  25.,   25.,    25.,   25.,   25.,  25.,   22.5,  12.5 ]])]

  for imgin, true_out in zip(expandin, expandout):
    if __name__ == "__main__":
      print "input:\n{}\n".format(imgin)

    usr_out = expand(imgin)

    if not type(usr_out) == type(true_out):
      if __name__ == "__main__":
        print "Error- expand out has type {}. Expected type is {}.".format(
            type(usr_out), type(true_out))
      return False

    if not usr_out.shape == true_out.shape:
      if __name__ == "__main__":
        print "Error- expand out has shape {}. Expected shape is {}.".format(
            usr_out.shape, true_out.shape)
      return False

    if not usr_out.dtype == true_out.dtype:
      if __name__ == "__main__":
        print "Error- expand out has dtype {}. Expected dtype is {}.".format(
            usr_out.dtype, true_out.dtype)
      return False

    if not np.all(np.abs(usr_out - true_out) < 1):
      if __name__ == "__main__":
        print "Error- expand out has value:\n{}\nExpected value:\n{}".format(
            usr_out, true_out)
      return False

  if __name__ == "__main__":
    print "expand passed."

  if __name__ == "__main__":
    print "All unit tests successful."
  return True

if __name__ == "__main__":
  print "Performing unit tests. Your functions will be accepted if your result is\
    within 1 of the correct output."
  np.set_printoptions(precision=1)

  test()

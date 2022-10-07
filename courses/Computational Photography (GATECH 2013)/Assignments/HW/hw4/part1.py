import numpy as np
import cv2
import scipy.signal

def binomial_filter_5():
  ''' Create a binomial filter of length 4.
  '''
  return np.array([ 0.0625,  0.25  ,  0.375 ,  0.25  ,  0.0625], dtype=float)

def diff2(diff1):
  '''Compute the transition costs between frames, taking dynamics into account.
  
  Input:
  diff1 - a difference matrix as produced by your ssd function.

  Output:
  output - a new difference matrix that takes preceding and following frames into 
           account. The cost of transitioning from i to j should weight the 
           surrounding frames according to the binomial filter provided to you 
           above. So, the difference between i and j has weight 0.375, the frames
           immediately following weight 0.25, etc...

           The output difference matrix should have the same dtype as the input,
           but be 4 rows and columns smaller, corresponding to only the frames
           that have valid dynamics.

  Hint: there is a very efficient way to do this with 2d convolution. Think about
        the coordinates you are using as you consider the preceding and following
        frame pairings.
  '''
  output = None
  # Insert your code here ------------------------------------------------------

  # ----------------------------------------------------------------------------
  return output

def test():
  '''This script will perform a unit test on your function, and provide useful
  output.
  '''
  d1 = np.zeros((9,9), dtype = float) 
  d1[4,4] = 1

  d2 = np.eye(5, dtype = float)

  out1 = np.array([[ 0.0625,  0.    ,  0.    ,  0.    ,  0.    ],
                   [ 0.    ,  0.25  ,  0.    ,  0.    ,  0.    ],
                   [ 0.    ,  0.    ,  0.375 ,  0.    ,  0.    ],
                   [ 0.    ,  0.    ,  0.    ,  0.25  ,  0.    ],
                   [ 0.    ,  0.    ,  0.    ,  0.    ,  0.0625]], dtype = float)

  out2 = np.array([[1.]], dtype = float)

  if __name__ == "__main__":
    print 'Evaluating diff2.'
  for d, true_out in zip((d1, d2), (out1, out2)):
    if __name__ == "__main__":
      print "input:\n{}\n".format(d)

    usr_out = diff2(d) 

    if not type(usr_out) == type(true_out):
      if __name__ == "__main__":
        print "Error- diff2 output has type {}. Expected type is {}.".format(
            type(usr_out), type(true_out))
      return False

    if not usr_out.shape == true_out.shape:
      if __name__ == "__main__":
        print "Error- diff2 output has shape {}. Expected shape is {}.".format(
            usr_out.shape, true_out.shape)
      return False

    if not usr_out.dtype == true_out.dtype:
      if __name__ == "__main__":
        print "Error- diff2 output has dtype {}. Expected dtype is {}.".format(
            usr_out.dtype, true_out.dtype)
      return False

    if not np.all(np.abs(usr_out - true_out) < 0.05):
      if __name__ == "__main__":
        print "Error- diff2 output has value:\n{}\nExpected value:\n{}".format(
            usr_out, true_out)
      return False

  if __name__ == "__main__":
    print "diff2 passed."

  if __name__ == "__main__":
    print "All unit tests successful."
  return True

if __name__ == "__main__":
  print "Performing unit tests. (Your output will be accepted if it is within .05 of the true output)"
  np.set_printoptions(precision=1)

  test()

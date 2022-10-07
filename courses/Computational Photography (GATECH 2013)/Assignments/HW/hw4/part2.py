import numpy as np
import cv2
import sys
import scipy.signal

def find_biggest_loop(diff2, alpha):
  '''Given the difference matrix, find the longest and smoothest loop that we can.
  
  Inputs:
  diff2 - a square 2d numpy array of dtype float. Each cell contains the cost
          of transitioning from frame i to frame j in the input video.

  alpha - a parameter for how heavily you should weigh the size of the loop
          relative to the transition cost of the loop. Larger alphas favor
          longer loops.

  Outputs:
  s - the beginning frame of the longest loop.
  f - the final frame of the longest loop.

  s, f will the indices in the diff2 matrix that give the maximum score
  according to the following metric:

  alpha*(f-s) - diff2[f,s]
  '''
  s = None
  f = None
  # Insert your code here ------------------------------------------------------

  # ----------------------------------------------------------------------------
  return s, f

def synthesize_loop(video_volume, s, f):
  '''Pull out the given loop from video.
  
  Input:
  video_volume - a (t, row, col, 3) array, as created by your video_volume function.
  i - the index of the starting frame.
  j - the index of the ending frame.

  Output:
  output - a list of arrays of size (row, col, 3) and dtype np.uint8, similar to
  the original input the video_volume function in part0.
  '''
  output = [] 
  # Insert your code here ------------------------------------------------------

  # ----------------------------------------------------------------------------
  return output

def test():
  '''This script will perform a unit test on your function, and provide useful
  output.
  '''
  d1 = np.ones((5,5), dtype = float)
  a1 = 1
  out1 = (0,4)

  d2 = np.array([[ 0.,  1.,  1.,  5.],
                 [ 1.,  0.,  3.,  4.],
                 [ 1.,  3.,  0.,  5.],
                 [ 5.,  4.,  5.,  0.]])
  a2 = 1 
  out2 = (0,2)

  d3 = np.array([[ 0.,  1.,  4.],
                 [ 1.,  0.,  1.],
                 [ 4.,  1.,  0.]])   
  a3 = 2
  out3 = (0,1)

  if __name__ == "__main__":
    print 'Evaluating find_biggest_loop'
  for d, a, true_out in zip((d1, d2, d3), (a1, a2, a3), (out1, out2, out3)):
    if __name__ == "__main__":
      print "input:\n{}\n".format(d)
      print "alpha = {}".format(a)

    usr_out = find_biggest_loop(d, a)

    if not usr_out == true_out:
      if __name__ == "__main__":
        print "Error- find_biggest_loop is {}. Expected output is {}.".format(
            usr_out, true_out)
      return False

  if __name__ == "__main__":
    print "find_biggest_loop passed."
    print "testing synthesize_loop."

  video_volume1 = np.array([[[[ 0,  0,  0],
                              [ 0,  0,  0]]],
                            [[[ 1,  1,  1],
                              [ 1,  1,  1]]],
                            [[[ 2,  2,  2],
                              [ 2,  2,  2]]],
                            [[[ 3,  3,  3],
                              [ 3,  3,  3]]]], dtype = np.uint8) 

  video_volume2 = np.array([[[[2, 2, 2],
                              [2, 2, 2],
                              [2, 2, 2],
                              [2, 2, 2]],
                             [[2, 2, 2],
                              [2, 2, 2],
                              [2, 2, 2],
                              [2, 2, 2]]],
                            [[[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]],
                             [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]]],
                            [[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]], dtype = np.uint8)

  frames1 = (2,3)
  frames2 = (1,1)


  out1 = [np.array([[[ 2,  2,  2],
                     [ 2,  2,  2]]], dtype = np.uint8),
          np.array([[[ 3,  3,  3],
                     [ 3,  3,  3]]], dtype = np.uint8)]

  out2 = [np.array([[[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]],
                    [[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]]], dtype = np.uint8)]

  for video_volume, frames, true_out in zip((video_volume1, video_volume2), 
      (frames1, frames2), (out1, out2)):
    if __name__ == "__main__":
      print "input:\n{}\n".format(video_volume)
      print "input frames:\n{}\n".format(frames)

    usr_out = synthesize_loop(video_volume, frames[0], frames[1])

    if not type(usr_out) == type(true_out):
      if __name__ == "__main__":
        print "Error- synthesize_loop has type {}. Expected type is {}.".format(
            type(usr_out), type(true_out))
      return False

    if not len(usr_out) == len(true_out):
      if __name__ == "__main__":
        print "Error - synthesize_loop has len {}. Expected len is {}.".format(
            len(usr_out), len(true_out))
        return False

    for usr_img, true_img in zip(usr_out, true_out):
      if not type(usr_img) == type(true_img):
        if __name__ == "__main__":
          print "Error- synthesize_loop has type {}. Expected type is {}.".format(
              type(usr_img), type(true_img))
        return False

      if not usr_img.shape == true_img.shape:
        if __name__ == "__main__":
          print "Error- synthesize_loop has shape {}. Expected shape is {}.".format(
              usr_img.shape, true_img.shape)
        return False

      if not usr_img.dtype == true_img.dtype:
        if __name__ == "__main__":
          print "Error- synthesize_loop has dtype {}. Expected dtype is {}.".format(
              usr_img.dtype, true_img.dtype)
        return False

      if not np.all(usr_img == true_img):
        if __name__ == "__main__":
          print "Error- synthesize_loop has value:\n{}\nExpected value:\n{}".format(
              usr_img, true_img)
        return False

  if __name__ == "__main__":
    print "synthesize_loop passed."

  if __name__ == "__main__":
    print "All unit tests successful."
  return True

if __name__ == "__main__":
  print "Performing unit tests."

  test()

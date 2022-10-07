import numpy as np
import cv2

def video_volume(image_list):
  '''Create a video volume from the image list.

  Input:
    image_list - a list of frames. Each element of the list contains a numpy
    array of a colored image. You may assume that each frame has the same shape,
    (rows, cols, 3).

  Output:
    output - a single 4d numpy array. This array should have dimensions
    (time, rows, cols, 3) and dtype np.uint8.
  '''
  output = None
  # Insert your code here ------------------------------------------------------

  # ----------------------------------------------------------------------------
  return output

def ssd(video_volume):
  '''Compute the sum of squared distances for each pair of frames in video volume.
  Input:
    video_volume - as returned by your video_volume function.

  Output:
    output - a square 2d numpy array of dtype float. output[i,j] should contain 
    the dsum of square differences between frames i and j.
  '''
  output = None
  # Insert your code here ------------------------------------------------------

  # ----------------------------------------------------------------------------
  return output

def test():
  '''This script will perform a unit test on your function, and provide useful
  output.
  '''
  image_list1 = [np.array([[[ 0,  0,  0],
                            [ 0,  0,  0]]], dtype = np.uint8),
                 np.array([[[ 1,  1,  1],
                            [ 1,  1,  1]]], dtype = np.uint8),
                 np.array([[[ 2,  2,  2],
                            [ 2,  2,  2]]], dtype = np.uint8),
                 np.array([[[ 3,  3,  3],
                            [ 3,  3,  3]]], dtype = np.uint8)] 

  video_volume1 = np.array([[[[ 0,  0,  0],
                              [ 0,  0,  0]]],
                            [[[ 1,  1,  1],
                              [ 1,  1,  1]]],
                            [[[ 2,  2,  2],
                              [ 2,  2,  2]]],
                            [[[ 3,  3,  3],
                              [ 3,  3,  3]]]], dtype = np.uint8) 

  image_list2 =[np.array([[[2, 2, 2],
                           [2, 2, 2],
                           [2, 2, 2],
                           [2, 2, 2]],
                          [[2, 2, 2],
                           [2, 2, 2],
                           [2, 2, 2],
                           [2, 2, 2]]], dtype = np.uint8),
                np.array([[[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]],
                          [[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]]], dtype = np.uint8),
                np.array([[[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]],
                          [[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]]], dtype = np.uint8)] 

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

  diff1 = np.array([[  0.,   6.,  24.,  54.],
                    [  6.,   0.,   6.,  24.],
                    [ 24.,   6.,   0.,   6.],
                    [ 54.,  24.,   6.,   0.]], dtype = np.float) 

  diff2 = np.array([[  0.,  24.,  96.],
                    [ 24.,   0.,  24.],
                    [ 96.,  24.,   0.]], dtype = np.float) 

  if __name__ == "__main__":
    print 'Evaluating video_volume.'
  for img_list, true_out in zip((image_list1, image_list2), (video_volume1, video_volume2)):
    if __name__ == "__main__":
      print "input:\n{}\n".format(img_list)

    usr_out = video_volume(img_list)

    if not type(usr_out) == type(true_out):
      if __name__ == "__main__":
        print "Error- video_volume has type {}. Expected type is {}.".format(
            type(usr_out), type(true_out))
      return False

    if not usr_out.shape == true_out.shape:
      if __name__ == "__main__":
        print "Error- video_volume has shape {}. Expected shape is {}.".format(
            usr_out.shape, true_out.shape)
      return False

    if not usr_out.dtype == true_out.dtype:
      if __name__ == "__main__":
        print "Error- video_volume has dtype {}. Expected dtype is {}.".format(
            usr_out.dtype, true_out.dtype)
      return False

    if not np.all(usr_out == true_out):
      if __name__ == "__main__":
        print "Error- video_volume has value:\n{}\nExpected value:\n{}".format(
            usr_out, true_out)
      return False

  if __name__ == "__main__":
    print "video_volume passed."
    print "evaluating ssd"

  for vid_volume, true_out in zip((video_volume1, video_volume2), (diff1, diff2)):
    if __name__ == "__main__":
      print "input:\n{}\n".format(vid_volume)

    usr_out = ssd(vid_volume)

    if not type(usr_out) == type(true_out):
      if __name__ == "__main__":
        print "Error- ssd has type {}. Expected type is {}.".format(
            type(usr_out), type(true_out))
      return False

    if not usr_out.shape == true_out.shape:
      if __name__ == "__main__":
        print "Error- ssd has shape {}. Expected shape is {}.".format(
            usr_out.shape, true_out.shape)
      return False

    if not usr_out.dtype == true_out.dtype:
      if __name__ == "__main__":
        print "Error- ssd has dtype {}. Expected dtype is {}.".format(
            usr_out.dtype, true_out.dtype)
      return False

    if not np.all(np.abs(usr_out - true_out) < 1.):
      if __name__ == "__main__":
        print "Error- ssd has value:\n{}\nExpected value:\n{}".format(
            usr_out, true_out)
      return False

  if __name__ == "__main__":
    print "All unit tests successful."
  return True

if __name__ == "__main__":
  print "Performing unit tests.(ssd will be accepted if the output is within 1 of the expected output.)"
  np.set_printoptions(precision=1)

  test()

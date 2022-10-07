import sys, os
import numpy as np
import cv2

def interlace(evens, odds):
  '''Reconstruct the image by alternating rows of evens and odds.

  evens - a numpy array of shape (rows, columns, 3)  containing the even rows 
          of the output image.
  odds - a numpy array of shape (rows, columns, 3) containing the odd rows 
         of the output image.

  This function should return an image. Row 0 of the output image should be
  row 0 of evens. Row 1 of the output image should be row 0 of odds. Then
  row 1 of evens, then row 1 of odds, and so on.

  The resulting image will have as many rows as image 1 and 2 combined, equal
  to both in number of columns, and have 3 channels.
  '''

  outimg = None
  # Implement your function here ---------------------------------------------

  #---------------------------------------------------------------------------
  return outimg

def main():
  ''' This code will attempt to reconstruct the images in the images/part1
  folder, and save the output.
  '''
  imagesfolder = os.path.join('images', 'part1')
  print "part 1 : attempting to interlace images evens.jpg and odds.jpg"

  evens = cv2.imread(os.path.join(imagesfolder, 'even.jpg'))
  odds = cv2.imread(os.path.join(imagesfolder, 'odd.jpg'))

  if evens == None or odds == None:
    print "Error - could not find even.jpg and odd.jpg in {}".format(imagesfolder)
    sys.exit(0)

  together = interlace(evens, odds)

  if not together == None:
    cv2.imwrite(os.path.join(imagesfolder, 'together.jpg'), together)

def test():
  '''This script will perform a unit test on your function, and provide useful
  output.
  '''
  x = (np.random.rand(4,4,3) * 255).astype(np.uint8)

  xeven = x[np.arange(0,x.shape[0], 2), :, :]
  xodd  = x[np.arange(1,x.shape[0], 2), :, :]

  if __name__== "__main__":
    print "Input:\n  even:\n{}\n  odd:\n{}".format(xeven, xodd)

  usr_out = interlace(xeven, xodd)

  if usr_out == None:
    if __name__ == "__main__":
      print "Error- output has value None."
    return False

  if not usr_out.shape == x.shape:
    if __name__ == "__main__":
      print "Error- output has shape {}. Expected shape is {}.".format(
          usr_out.shape, x.shape)
    return False

  if not usr_out.dtype == x.dtype:
    if __name__ == "__main__":
      print "Error- output has dtype {}. Expected dtype is {}.".format(
          usr_out.dtype, x.dtype)
    return False

  if not np.all(usr_out == x):
    if __name__ == "__main__":
      print "Error- output has value:\n{}\nExpected value:\n{}".format(
          usr_out, x)
    return False

  if __name__ == "__main__":
    print "Success - all outputs correct."
  return True

if __name__ == "__main__":
  # Testing code
  t = test()
  print "Unit test: {}".format(t)
  if t:
    main()

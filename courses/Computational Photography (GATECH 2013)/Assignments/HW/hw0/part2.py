import sys
import os
import numpy as np
import cv2

def greyscale(image):
  '''Convert an image to greyscale.
  
  image  - a numpy array of shape (rows, columns, 3).
  output - a numpy array of shape (rows, columns) and dtype same as
           image, containing the average of image's 3 channels. 
  
  Please make sure the output shape has only 2 components!
  For instance, (512, 512) instead of (512, 512, 1)
  '''
  output = None
  # Insert your code here.----------------------------------------------------

  #---------------------------------------------------------------------------
  return output

def main():
  '''Convert images to greyscale.

  It will search through the images/part2 subfolder, and apply your function
  to each one, saving the output image in the same folder.
  '''

  imagesfolder = os.path.join('images', 'part2')

  print '''part 2 : Searching for images in {} folder
  (will ignore if grey in the name)'''.format(imagesfolder)

  exts = ['.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpeg', '.jpg', 
    '.jpe', '.jp2', '.tiff', '.tif', '.png']

  for dirname, dirnames, filenames in os.walk(imagesfolder):
    for filename in filenames:
      name, ext = os.path.splitext(filename)
      if ext in exts and 'grey' not in name:
        print "Attempting to split image {}.".format(filename)

        img = cv2.imread(os.path.join(dirname, filename))
        grey = greyscale(img)
        print "Writing image {}".format(name+"grey"+ext)
        cv2.imwrite(os.path.join(dirname, name+"grey"+ext), grey)

def test():
  '''This script will perform a unit test on your function, and provide useful
  output.
  '''
  x = (np.random.rand(4,4,3) * 255).astype(np.uint8)

  if __name__ == "__main__":
    print "Input:\n{}".format(x)

  usr_grey = greyscale(x)
  true_grey = (x.sum(2)/3).astype(np.uint8)

  if usr_grey == None:
    if __name__ == "__main__":
      print "Error- output has value None."
    return False

  if not usr_grey.shape == true_grey.shape:
    if __name__ == "__main__":
      print "Error- output has shape {}. Expected shape is {}.".format(
          usr_grey.shape, true_grey.shape)
    return False

  if not usr_grey.dtype == true_grey.dtype:
    if __name__ == "__main__":
      print "Error- output has dtype {}. Expected dtype is {}.".format(
          usr_grey.dtype, true_grey.dtype)
    return False

  if not np.all(usr_grey == true_grey):
    if __name__ == "__main__":
      print "Error- output has value:\n{}\nExpected value:\n{}".format(
          usr_grey, true_grey)
    return False

  if __name__ == "__main__":
    print "Success - all outputs correct."
  return True

if __name__ == "__main__":
  # Testing code
  t = test()
  print "Unit test - {}".format(t)
  if t:
    main()

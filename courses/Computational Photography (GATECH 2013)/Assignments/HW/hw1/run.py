import sys
import os
import numpy as np
import cv2
from scipy import signal
import math

import part0
import part1
import part2
import part3

def box_filter(k=1):
  '''Return a box filter of shape (2k+1, 2k+1) and dtype float.
  '''
  return np.ones((2*k+1, 2*k+1), dtype = float)/((2*k+1)**2)

def apply_median(k):
  ''' Apply the given kernel to images

  This function searches through the images/source subfolder, and
  uses your convolution funciton implemented in part0 to apply the given kernel
  to each image found inside. It will then save the resulting images to the 
  images/filtered subfolder, appending their names with kernel_name.
  '''
  print 'applying median filter to images'

  sourcefolder = os.path.abspath(os.path.join(os.curdir, 'images', 'source'))
  outfolder = os.path.abspath(os.path.join(os.curdir, 'images', 'filtered'))

  print 'Searching for images in {} folder'.format(sourcefolder)

  exts = ['.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpeg', '.jpg', 
    '.jpe', '.jp2', '.tiff', '.tif', '.png']

  for dirname, dirnames, filenames in os.walk(sourcefolder):
    for filename in filenames:
      name, ext = os.path.splitext(filename)
      if ext in exts:
        print "Reading image {}.".format(filename)
        img = cv2.imread(os.path.join(dirname, filename))

        print "Applying filter."
        if len(img.shape) == 2:
          outimg = part3.filter_median(img, k)
        else:
          outimg = [] 
          for channel in range(img.shape[2]):
            outimg.append(part3.filter_median(img[:,:,channel], k))
          outimg = cv2.merge(outimg)
        outpath = os.path.join(outfolder, name + 'median' + str(k) + ext)

        print "Writing image {}.\n\n".format(outpath)
        cv2.imwrite(outpath, outimg)

def apply_filter(conv_func, kernel, kernel_name):
  ''' Apply the given kernel to images

  This function searches through the images/source subfolder, and
  uses your convolution funciton implemented in part0 to apply the given kernel
  to each image found inside. It will then save the resulting images to the 
  images/filtered subfolder, appending their names with kernel_name.
  '''
  print 'applying {} kernel to images'.format(kernel_name)

  sourcefolder = os.path.abspath(os.path.join(os.curdir, 'images', 'source'))
  outfolder = os.path.abspath(os.path.join(os.curdir, 'images', 'filtered'))

  print 'Searching for images in {} folder'.format(sourcefolder)

  exts = ['.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpeg', '.jpg', 
    '.jpe', '.jp2', '.tiff', '.tif', '.png']

  for dirname, dirnames, filenames in os.walk(sourcefolder):
    for filename in filenames:
      name, ext = os.path.splitext(filename)
      if ext in exts:
        print "Reading image {}.".format(filename)
        img = cv2.imread(os.path.join(dirname, filename))

        print "Applying filter."
        if len(img.shape) == 2:
          outimg = conv_func(img, kernel)
        else:
          outimg = []
          for channel in range(img.shape[2]):
            outimg.append(conv_func(img[:,:,channel], kernel))
          outimg = cv2.merge(outimg)
        outpath = os.path.join(outfolder, name + kernel_name + ext)

        print "Writing image {}.\n\n".format(outpath)
        cv2.imwrite(outpath, outimg)

if __name__ == "__main__":
  # Testing code --------------------------------------------------------------
  # feel free to modify this to try out different filters and parameters.

  print "-"*15 + "part0" + "-"*15
  t0 = part0.test()
  print "Unit test: {}".format(t0)
  conv_func = signal.convolve2d 
  if t0:
    conv_func = part0.convolve
    apply_filter(conv_func, box_filter(2), 'box2')
  else:
    print "Please test your code using part0.py prior to using this function."

  print "-"*15 + "part1" + "-"*15
  t1 = part1.test()
  print "Unit test: {}".format(t1)
  if t1:
    apply_filter(conv_func, part1.make_gaussian(5,3), 'gaussian5_3')
  else:
    print "Please test your code using part1.py prior to using this function."

  print "-"*15 + "part2" + "-"*15
  t2 = part2.test()
  print "Unit test: {}".format(t2)
  if t2:
    apply_filter(conv_func, part2.make_sharp(5,3), 'sharp5_3')
  else:
    print "Please test your code using part2.py prior to using this function."

  print "-"*15 + "part3" + "-"*15
  t3 = part3.test()
  print "Unit test: {}".format(t3)
  if t3:
    apply_median(5)
  else:
    print "Please test your code using part3.py prior to using this function."



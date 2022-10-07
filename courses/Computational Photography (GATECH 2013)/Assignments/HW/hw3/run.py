import sys
import os
import numpy as np
import cv2
from scipy.stats import norm
from scipy.signal import convolve2d
import math

import part0
import part1
import part2


def viz_gauss_pyramid(pyramid):
  ''' This function creates a single image out of the given pyramid.
  '''
  height = pyramid[0].shape[0]
  width = pyramid[0].shape[1]

  out = np.zeros((height*len(pyramid), width), dtype = float)

  for idx, layer in enumerate(pyramid):
    if layer.max() <= 1:
      layer = layer.copy() * 255

    out[(idx*height):((idx+1)*height),:] = cv2.resize(layer, (width, height), 
        interpolation=cv2.cv.CV_INTER_AREA)

  return out.astype(np.uint8)

def viz_lapl_pyramid(pyramid):
  ''' This function creates a single image out of the given pyramid.
  '''
  height = pyramid[0].shape[0]
  width = pyramid[0].shape[1]

  out = np.zeros((height*len(pyramid), width), dtype = np.uint8)

  for idx, layer in enumerate(pyramid[:-1]):
     patch = cv2.resize(layer, (width, height), 
         interpolation=cv2.cv.CV_INTER_AREA).astype(float)
     # scale patch to 0:256 range.
     patch = 128 + 127*patch/(np.abs(patch).max())

     out[(idx*height):((idx+1)*height),:] = patch.astype(np.uint8)

  #special case for the last layer, which is simply the remaining image.
  patch = cv2.resize(pyramid[-1], (width, height), 
      interpolation=cv2.cv.CV_INTER_AREA)
  out[((len(pyramid)-1)*height):(len(pyramid)*height),:] = patch

  return out

def run_blend(black_image, white_image, mask):
  ''' This function administrates the blending of the two images according to 
  mask.

  Assume all images are float dtype, and return a float dtype.
  '''

  # Automatically figure out the size
  min_size = min(black_image.shape)
  depth = int(math.floor(math.log(min_size, 2))) - 4 # at least 16x16 at the highest level.

  gauss_pyr_mask = part1.gauss_pyramid(mask, depth)
  gauss_pyr_black = part1.gauss_pyramid(black_image, depth)
  gauss_pyr_white = part1.gauss_pyramid(white_image, depth)


  lapl_pyr_black  = part1.lapl_pyramid(gauss_pyr_black)
  lapl_pyr_white = part1.lapl_pyramid(gauss_pyr_white)

  outpyr = part2.blend(lapl_pyr_black, lapl_pyr_white, gauss_pyr_mask)
  outimg = part2.collapse(outpyr)

  outimg[outimg<0] = 0 # blending sometimes results in slightly out of bound numbers.
  outimg[outimg>255] = 255
  outimg = outimg.astype(np.uint8)

  return lapl_pyr_black, lapl_pyr_white, gauss_pyr_black, gauss_pyr_white, \
      gauss_pyr_mask, outpyr, outimg

if __name__ == "__main__":
  print 'Performing unit tests.'
  if not part0.test():
    print 'part0 failed. halting'
    sys.exit()

  if not part1.test():
    print 'part1 failed. halting'
    sys.exit()

  if not part2.test():
    print 'part2 failed. halting'
    sys.exit()

  print 'Unit tests passed.'
  sourcefolder = os.path.abspath(os.path.join(os.curdir, 'images', 'source'))
  outfolder = os.path.abspath(os.path.join(os.curdir, 'images', 'output'))

  print 'Searching for images in {} folder'.format(sourcefolder)

  # Extensions recognized by opencv
  exts = ['.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpeg', '.jpg', 
    '.jpe', '.jp2', '.tiff', '.tif', '.png']

  # For every image in the source directory
  for dirname, dirnames, filenames in os.walk(sourcefolder):
    setname = os.path.split(dirname)[1]

    white_img = None
    black_img = None
    mask_img = None

    for filename in filenames:
      name, ext = os.path.splitext(filename)
      if ext in exts:
        if 'black' in name:
          print "Reading image {}.".format(filename)
          black_img = cv2.imread(os.path.join(dirname, filename))

        if 'white' in name:
          print "Reading image {}.".format(filename)
          white_img = cv2.imread(os.path.join(dirname, filename))

        if 'mask' in name:
          print "Reading image {}.".format(filename)
          mask_img = cv2.imread(os.path.join(dirname, filename))

    if white_img == None or black_img == None or mask_img == None:
      print "Not all images found. Skipping."
      continue

    assert black_img.shape == white_img.shape and black_img.shape == mask_img.shape, \
        "Error - the sizes of images and the mask are not equal"

    black_img = black_img.astype(float)
    white_img = white_img.astype(float)
    mask_img = mask_img.astype(float) / 255

    print "Applying blending."
    lapl_pyr_black_layers = []
    lapl_pyr_white_layers = []
    gauss_pyr_black_layers = []
    gauss_pyr_white_layers = []
    gauss_pyr_mask_layers = []
    out_pyr_layers = []
    out_layers = []

    for channel in range(3):
      lapl_pyr_black, lapl_pyr_white, gauss_pyr_black, gauss_pyr_white, gauss_pyr_mask,\
          outpyr, outimg = run_blend(black_img[:,:,channel], white_img[:,:,channel], \
                           mask_img[:,:,channel])
      
      lapl_pyr_black_layers.append(viz_lapl_pyramid(lapl_pyr_black))
      lapl_pyr_white_layers.append(viz_lapl_pyramid(lapl_pyr_white))
      gauss_pyr_black_layers.append(viz_gauss_pyramid(gauss_pyr_black))
      gauss_pyr_white_layers.append(viz_gauss_pyramid(gauss_pyr_white))
      gauss_pyr_mask_layers.append(viz_gauss_pyramid(gauss_pyr_mask))
      out_pyr_layers.append(viz_lapl_pyramid(outpyr))
      out_layers.append(outimg)
    
    lapl_pyr_black_img = cv2.merge(lapl_pyr_black_layers)
    lapl_pyr_white_img = cv2.merge(lapl_pyr_white_layers)
    gauss_pyr_black_img = cv2.merge(gauss_pyr_black_layers)
    gauss_pyr_white_img = cv2.merge(gauss_pyr_white_layers)
    gauss_pyr_mask_img = cv2.merge(gauss_pyr_mask_layers)
    outpyr = cv2.merge(out_pyr_layers)
    outimg = cv2.merge(out_layers)

    print "Writing images to folder {}".format(os.path.join(outfolder, setname))
    cv2.imwrite(os.path.join(outfolder, setname+'_lapl_pyr_black'+ext), lapl_pyr_black_img)
    cv2.imwrite(os.path.join(outfolder, setname+'_lapl_pyr_white'+ext), lapl_pyr_white_img)
    cv2.imwrite(os.path.join(outfolder, setname+'_gauss_pyr_black'+ext), gauss_pyr_black_img)
    cv2.imwrite(os.path.join(outfolder, setname+'_gauss_pyr_white'+ext), gauss_pyr_white_img)
    cv2.imwrite(os.path.join(outfolder, setname+'_gauss_pyr_mask'+ext), gauss_pyr_mask_img)
    cv2.imwrite(os.path.join(outfolder, setname+'_outpyr'+ext), outpyr)
    cv2.imwrite(os.path.join(outfolder, setname+'_outimg'+ext), outimg)

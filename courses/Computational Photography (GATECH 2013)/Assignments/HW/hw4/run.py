import sys
import os
import numpy as np
import cv2

import part0
import part1
import part2

def viz_diff(diff):
  return (((diff-diff.min())/(diff.max()-diff.min()))*255).astype(np.uint8)

def run_texture(img_list):
  ''' This function administrates the extraction of a video texture from the given
  frames.'''
  video_volume = part0.video_volume(img_list)
  diff1 = part0.ssd(video_volume)
  diff2 = part1.diff2(diff1)
  alpha = 1.5*10**6
  idxs = part2.find_biggest_loop(diff2, alpha)

  diff3 = np.zeros(diff2.shape, float)

  for i in range(diff2.shape[0]): 
    for j in range(diff2.shape[1]): 
      diff3[i,j] = alpha*(i-j) - diff2[i,j] 

  return viz_diff(diff1), viz_diff(diff2), viz_diff(diff3),\
    part2.synthesize_loop(video_volume, idxs[0]+2, idxs[1]+2)

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
  sourcefolder = os.path.abspath(os.path.join(os.curdir, 'videos', 'source'))
  outfolder = os.path.abspath(os.path.join(os.curdir, 'videos', 'out'))

  print 'Searching for video folders in {} folder'.format(sourcefolder)

  # Extensions recognized by opencv
  exts = ['.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpeg', '.jpg', 
    '.jpe', '.jp2', '.tiff', '.tif', '.png']

  # For every image in the source directory
  for viddir in os.listdir(sourcefolder):
    print "collecting images from directory {}".format(viddir)
    img_list = []
    filenames = sorted(os.listdir(os.path.join(sourcefolder, viddir)))

    for filename in filenames:
      name, ext = os.path.splitext(filename)
      if ext in exts:
        img_list.append(cv2.imread(os.path.join(sourcefolder, viddir, filename)))
    
    print "extracting video texture frames."
    diff1, diff2, diff3, out_list = run_texture(img_list)

    cv2.imwrite(os.path.join(outfolder, '{}diff1.png'.format(viddir)), diff1)
    cv2.imwrite(os.path.join(outfolder, '{}diff2.png'.format(viddir)), diff2)
    cv2.imwrite(os.path.join(outfolder, '{}diff3.png'.format(viddir)), diff3)

    print "writing output to {}".format(os.path.join(outfolder, viddir))
    if not os.path.exists(os.path.join(outfolder, viddir)):
      os.mkdir(os.path.join(outfolder, viddir))

    for idx, image in enumerate(out_list):
      cv2.imwrite(os.path.join(outfolder,viddir,'frame{0:04d}.png'.format(idx)), image)


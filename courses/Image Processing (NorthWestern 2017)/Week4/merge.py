# This script will take a folder containing video frames and merge them together
# into a video.

import argparse
import cv2
import sys, os

parser = argparse.ArgumentParser(description='Merge frames into video.')
parser.add_argument('img_dir', type=str, help='folder containing image frames')
args = parser.parse_args()

if not os.path.exists(args.img_dir):
  print "Error - the given path is not valid: {}".format(args.img_dir)

writer = cv2.VideoWriter()

filenames = sorted(os.listdir(args.img_dir))

exts = ['.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpeg', '.jpg', 
    '.jpe', '.jp2', '.tiff', '.tif', '.png']

for filename in filenames:
  name, ext = os.path.splitext(filename)
  if ext in exts:
    img = cv2.imread(os.path.join(args.img_dir, filename))
    if not writer.isOpened():
      writer.open(os.path.join(args.img_dir, 'video.avi'), 
          cv2.cv.CV_FOURCC('I','4','2','0'), 30, (img.shape[0], img.shape[1]))
    writer.write(img)

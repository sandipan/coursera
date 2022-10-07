# This script will split the target video file into frames in a subfolder
# under the same name.

import argparse
import cv2
import sys, os

parser = argparse.ArgumentParser(description='Split the target video into frames.')
parser.add_argument('vid_path', type=str, help='the path to the target video.')
args = parser.parse_args()

if not os.path.exists(args.vid_path):
  print "Error - the given path is not valid: {}".format(args.vid_path)

cap = cv2.VideoCapture(args.vid_path)

vid_dir, vid_filename = os.path.split(args.vid_path)
vid_name, vid_ext = os.path.splitext(vid_filename)

out_dir = os.path.join(vid_dir, vid_name)

if not os.path.exists(out_dir):
  os.mkdir(out_dir)

count = 0
while cap.grab():
  success, img = cap.retrieve()
  if not success:
    break
  cv2.imwrite(os.path.join(out_dir, 'frame{0:04d}.png'.format(count)), img)
  count += 1

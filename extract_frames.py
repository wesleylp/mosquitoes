import numpy as np
import cv2
import os
print('OpenCV version %s' % cv2.__version__)

#data path
vid_path = '../videos/'
vid_name = 'DJI00879'
vid_format = 'MP4'

#path to save the frames
frames_path = vid_path + vid_name + '/'
#create fold to save the frame if it doesn't exist
if not os.path.exists(frames_path):
    os.makedirs(frames_path)

#load video
vidcap = cv2.VideoCapture(vid_path + vid_name + '.' + vid_format)
success, image = vidcap.read()
count = 0
success = True
while success:
    # save frame as PNG file
    cv2.imwrite(frames_path + 'frame%d.png' % count, image)
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
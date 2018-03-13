import numpy as np
import cv2
import os
print('OpenCV version %s' % cv2.__version__)

#data path
vid_path = 'data/'
vid_name = 'DJI00884'
vid_format = 'MP4'

#path to save the frames
frames_path = vid_path+'frames_'+vid_name+'/'
print(frames_path)

#create fold to save the frame if it doesn't exist
if not os.path.exists(frames_path):
    os.makedirs(frames_path)

#load video
video_capture = cv2.VideoCapture(vid_path+vid_name+'.'+vid_format)
video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
print('video has %d frames' % video_length)
success,image = video_capture.read()
count = 0
success = True
while success:
    cv2.imwrite(frames_path+'frame%d.jpeg' % count, image)     # save frame as png file
    success,image = video_capture.read()
    print('Reading frame %d: '% count,  success)
    count += 1
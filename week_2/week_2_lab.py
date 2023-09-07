'''
Tom Mulroy
CS549
Lab 1
NOTE: Some code used from OpenCV documentation:
https://docs.opencv.org/4.x/dc/d4d/tutorial_py_table_of_contents_gui.html
'''

import cv2 as cv
import numpy as np
from datetime import datetime
import time

# Video Capture
cap = cv.VideoCapture(0)
cv.namedWindow('Video')

# Flags
record_video_flag = False
extract_color_flag = False
rotate_image_flag = False
threshold_image_flag = False
screenshot_flag = False
blur_flag = False
sharpen_flag = False

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('video_output.avi', fourcc, 20.0, (640, 480))

if not cap.isOpened():
 print("Cannot open camera")
 exit()

def handle_trackbar(value):
 pass

cv.createTrackbar('sigmaX','Video',5,30,handle_trackbar)
cv.createTrackbar('sigmaY','Video',5,30,handle_trackbar)

# Main Loop
while cap.isOpened():

 # Capture frame-by-frame
 ret, frame = cap.read()

 # if frame is read correctly ret is True
 if not ret:
  print("Can't receive frame (stream end?). Exiting ...")
  break

 # Our operations on the frame come here
 frame = cv.flip(frame,1)

 # ADD BORDER
 frame = cv.copyMakeBorder(frame, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=[0, 0, 255])

# TIMESTAMP
 # Get frame info to display text
 font = cv.FONT_HERSHEY_PLAIN
 rows = frame.shape[0]
 cols = frame.shape[1]
 timestamp_x_location = int(round(1.5 * (cols / 3)))
 timestamp_y_location = int(round(2.75 * (rows / 3)))

 cv.putText(frame, str(datetime.now()), (timestamp_x_location, timestamp_y_location), cv.FONT_HERSHEY_PLAIN, 2,
            (255, 255, 255), 2, cv.LINE_AA)

 # REGION OF INTEREST
 timestamp_roi = frame[430:460, 320:640]
 frame[10:40, 320:640] = timestamp_roi

 # ADD OPENCV LOGO
 logo_file = cv.imread('opencv_logo.png')
 logo = cv.resize(logo_file, (0, 0), fx=0.5, fy=0.5)
 if logo is None:
  sys.exit('Could not read file')
 logo_rows = logo.shape[0]
 logo_cols = logo.shape[1]

 src1 = frame[10:logo_rows + 10, 10:logo_cols + 10]

 alpha = 0.5
 beta = 1-alpha

 frame[10:logo_rows + 10, 10:logo_cols + 10] = cv.addWeighted(src1,alpha,logo,beta, 0.0)

 # EXTRACT PINK COLOR
 if extract_color_flag == True:
  print('extracting color')

  # Convert BGR to HSV
  hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

  # define range of blue color in HSV
  lower_pink = np.array([140, 50, 50])
  upper_pink = np.array([160, 255, 255])
  # Threshold the HSV image to get only blue colors
  mask = cv.inRange(hsv, lower_pink, upper_pink)
  # Bitwise-AND mask and original image
  res = cv.bitwise_and(frame, frame, mask=mask)
  frame[:,:] = res

 # ROTATE IMAGE
 if rotate_image_flag == True:
  M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 10, 0.8)
  frame = cv.warpAffine(frame, M, (cols, rows))

 # THRESHOLD
 if threshold_image_flag == True:
  frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  ret,thresh = cv.threshold(frame,127,255,cv.THRESH_BINARY)
  frame = thresh

 # GAUSSIAN BLUR
 if blur_flag == True:
  sigmaX = cv.getTrackbarPos('sigmaX','Video')
  sigmaY = cv.getTrackbarPos('sigmaY','Video')
  blur = cv.GaussianBlur(frame, (9, 9), sigmaX)
  frame = blur

 # SHARPEN
 if sharpen_flag == True:
  blur = cv.GaussianBlur(frame,(5,5),20)
  frame = cv.addWeighted(frame,2,blur,-1,0,0)

 # SCREENSHOT
 if screenshot_flag == True:
  screenshot = frame[:,:]
  cv.imwrite('screenshot.png', screenshot)
  if threshold_image_flag == True:
   frame[:,:] = np.full((500,660),255,dtype=int)
  else:
   frame[:,:] = [255,255,255]
  screenshot_flag = False

 # Display Frame
 cv.imshow('Video',frame)

 # Save video only if 'v' key is pressed again
 if record_video_flag == True:
  print('saving video')
  out.write(frame)

 # Handle Keyboard Inputs
 k = cv.waitKeyEx(1)
 if k == 27: # Exit Program
  print('Exiting program')
  break

 elif k == ord('c'): # Take a Screenshot
  print('pressed c')
  if screenshot_flag == True:
   screenshot_flag = False
  else:
   screenshot_flag = True

 elif k == ord('v'): # Handle Video Recording Logic
  if record_video_flag == True:
   print('stopped recording video')
   record_video_flag = False
  else:
   print('recording video')
   record_video_flag = True

 elif k == ord('e'): # Extract a color
  print('pressed e')
  if extract_color_flag == False:
   extract_color_flag = True
  else:
   extract_color_flag = False

 elif k == ord('r'): # Rotate image
  print('pressed r')
  if rotate_image_flag == False:
   rotate_image_flag = True
  else:
   rotate_image_flag = False

 elif k == ord('t'):
  print('pressed t')
  if threshold_image_flag == False:
   threshold_image_flag = True
  else:
   threshold_image_flag = False

 elif k == ord('b'):
  print('pressed b')
  if blur_flag == False:
   blur_flag = True
  else:
   blur_flag = False
 elif k == ord('s'):
  print('pressed s')
  if sharpen_flag == False:
   sharpen_flag = True
  else:
   sharpen_flag = False

# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()
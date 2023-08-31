'''
Tom Mulroy
CS549
Lab 1
NOTE: Some code used from OpenCV documentation:
https://docs.opencv.org/4.x/dc/d4d/tutorial_py_table_of_contents_gui.html
'''

import cv2 as cv
from datetime import datetime

# Video Capture
cap = cv.VideoCapture(0)

record_video_flag = False

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('video_output.avi', fourcc, 20.0, (640, 480))

if not cap.isOpened():
 print("Cannot open camera")
 exit()

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

 # Display the resulting frame
 cv.imshow('frame',frame)

 # Get frame info to display text
 font = cv.FONT_HERSHEY_PLAIN
 height = frame.shape[0]
 width = frame.shape[1]
 timestamp_x_location = int(round(1.5 * (width / 3)))
 timestamp_y_location = int(round(2.75 * (height / 3)))

 # Save video only if 'v' key is pressed again
 if record_video_flag == True:
  print('saving video')
  cv.putText(frame, str(datetime.now()), (timestamp_x_location, timestamp_y_location), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv.LINE_AA)
  out.write(frame)

 # Handle Keyboard Input
 k = cv.waitKeyEx(1)
 if k == 27: # Exit Program
  print('Exiting program')
  break

 elif k == ord('c'): # Take a Screenshot
  print('taking a screenshot')
  cv.putText(frame, str(datetime.now()), (timestamp_x_location, timestamp_y_location), font, 2, (255, 255, 255), 2,
             cv.LINE_AA)
  cv.imwrite('screenshot.png', frame)

 elif k == ord('v'): # Handle Video Recording Logic
  if record_video_flag == True:
   print('stopped recording video')
   record_video_flag = False
  else:
   print('recording video')
   record_video_flag = True

# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()
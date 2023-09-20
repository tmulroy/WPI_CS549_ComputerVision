'''
Tom Mulroy
CS549
Lab 3
NOTE: Some code used from OpenCV documentation
'''
import cv2
import cv2 as cv
import numpy as np
from datetime import datetime
import time
from sobel import custom_sobel
from laplacian import custom_laplacian

# Video Capture
cap = cv.VideoCapture(0)
cv.namedWindow('Original')

# Instantiate sift for sift feature detection
sift = cv.SIFT_create()

# Flags
record_video_flag = False
extract_color_flag = False
rotate_image_flag = False
threshold_image_flag = False
screenshot_flag = False
blur_flag = False
sharpen_flag = False
x_key_flag = False
y_key_flag = False
s_key_flag = False
canny_flag = False
custom_ops_flag = False
harris_detection_flag = False
sift_detection_flag = False

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('video_output.avi', fourcc, 20.0, (640, 480))

if not cap.isOpened():
 print("Cannot open camera")
 exit()


def handle_trackbar(value):
 pass


# TRACKBARS
cv.createTrackbar('sigmaX', 'Original', 5, 30, handle_trackbar)
cv.createTrackbar('sigmaY', 'Original', 5, 30, handle_trackbar)
cv.createTrackbar('Sobel X', 'Original', 5, 30, handle_trackbar)
cv.createTrackbar('Sobel Y', 'Original', 5, 30, handle_trackbar)
cv.createTrackbar('Canny Threshold 1', 'Original', 1, 5000, handle_trackbar)
cv.createTrackbar('Canny Threshold 2', 'Original', 1, 5000, handle_trackbar)


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

  # Convert BGR to HSV
  hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

  # define range of pink color in HSV
  lower_pink = np.array([140, 50, 50])
  upper_pink = np.array([160, 255, 255])
  # Threshold the HSV image to get only pink colors
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
  sigmaX = cv.getTrackbarPos('sigmaX','Original')
  sigmaY = cv.getTrackbarPos('sigmaY','Original')
  blur = cv.GaussianBlur(frame, (9, 9), sigmaX)
  frame = blur

 # SHARPEN
 if s_key_flag == True and x_key_flag == False and y_key_flag == False:
  blur = cv.GaussianBlur(frame,(5,5),20)
  frame = cv.addWeighted(frame,2,blur,-1,0,0)

# SOBEL OPERATORS
 if s_key_flag == True:
  if x_key_flag == True:
   y_key_flag == False
   kernel_size = cv.getTrackbarPos('Sobel X', 'Original')
   if kernel_size % 2 == 0:
    kernel_size -= 1
   blurred = cv.GaussianBlur(frame, (3, 3), 50)
   gray = cv.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   frame = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=kernel_size)
  elif y_key_flag == True:
   x_key_flag == False
   kernel_size = cv.getTrackbarPos('Sobel Y', 'Original')
   if kernel_size % 2 == 0:
    kernel_size -= 1
   blurred = cv.GaussianBlur(frame, (3, 3), 50)
   gray = cv.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   frame = cv.Sobel(gray, cv.CV_64F, 0 , 1, ksize=kernel_size)

# CANNY EDGE DETECTOR
 if canny_flag == True:
  min_val = cv.getTrackbarPos('Canny Threshold 1', 'Original')
  max_val = cv.getTrackbarPos('Canny Threshold 2', 'Original')
  frame = cv.Canny(frame, min_val, max_val, True)

# CUSTOM LAPLACE AND SOBEL OPERATORS
 if custom_ops_flag == True:
  blurred = cv.GaussianBlur(frame, (9, 9), 1)
  gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
  cv.namedWindow('Sobel X')
  cv.namedWindow('Sobel Y')
  cv.namedWindow('Laplacian')
  sobel_x_frame = custom_sobel(gray, dx=1, dy=0)
  sobel_y_frame = custom_sobel(gray, dx=0, dy=1)
  # laplacian = cv.Laplacian(gray, ddepth=-1, ksize=3)
  laplacian = custom_laplacian(gray)
  cv.imshow('Sobel X', sobel_x_frame)
  cv.imshow('Sobel Y', sobel_y_frame)
  cv.imshow('Laplacian', laplacian)

# Harris Corner Detection
 if harris_detection_flag == True:
  gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  gray_float32 = np.float32(gray)
  corners = cv.cornerHarris(gray_float32, 3, 5, 0.04)
  dilated = cv.dilate(corners, None)
  frame[dilated > 0.04 * dilated.max()] = [0, 0, 255]

# SIFT Feature Detection
 if sift_detection_flag == True:
  gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  kp = sift.detect(gray, None)
  frame = cv.drawKeypoints(gray, kp, frame)

 # SCREENSHOT
 if screenshot_flag == True:
  screenshot = frame[:,:]
  cv.imwrite('screenshot.png', screenshot)
  if threshold_image_flag or (s_key_flag and x_key_flag) or (s_key_flag and y_key_flag) or (canny_flag==True) == True:
   frame[:,:] = np.full((500,660),255,dtype=int)
  else:
   frame[:,:] = [255,255,255]
  screenshot_flag = False

 # Save video only if 'v' key is pressed again
 if record_video_flag == True:
  print('saving video')
  out.write(frame)

 # Display Frame
 cv.imshow('Original',frame)

 # Handle Keyboard Inputs
 k = cv.waitKeyEx(1)
 # if k != -1:
 #  print(f'k: {k}')

 if k == 27: # Exit Program
  print('Exiting program')
  break

 elif k == ord('c'): # Take a Screenshot
  print('pressed c')
  screenshot_flag = not screenshot_flag

 elif k == ord('v'): # Handle Video Recording Logic
  if record_video_flag == True:
   print('stopped recording video')
   record_video_flag = False
  else:
   print('recording video')
   record_video_flag = True

 elif k == ord('e'): # Extract a color
  print('pressed e')
  extract_color_flag = not extract_color_flag

 elif k == ord('r'): # Rotate image
  print('pressed r')
  rotate_image_flag = not rotate_image_flag

 elif k == ord('t'): # Threshold Image
  print('pressed t')
  threshold_image_flag = not threshold_image_flag

 elif k == ord('b'): # Blur Image
  print('pressed b')
  blur_flag = not blur_flag

 elif k == ord('s'):
  print('pressed s')
  s_key_flag = not s_key_flag

 elif k == ord('x'): # Sobel X
  print('pressed x')
  if x_key_flag == True:
   s_key_flag = False
   x_key_flag = False
  else:
   x_key_flag = True
 elif k == ord('y'): # Sobel Y
  print('pressed y')
  if y_key_flag == True:
   s_key_flag = False
   y_key_flag = False
  else:
   y_key_flag = True

 elif k == ord('d'): # Canny Edge Detector
  print('pressed d')
  canny_flag = not canny_flag

 elif k == ord('4'): # Custom Operations
  print('pressed 4')
  custom_ops_flag = not custom_ops_flag

 elif k == ord('h'):
  print('pressed h for harris corner detection')
  harris_detection_flag = not harris_detection_flag

 elif k == ord('f'):
  sift_detection_flag = not sift_detection_flag


# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()
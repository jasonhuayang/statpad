import cv2
import numpy as np

VIDEO_PATH = 'samples/sample.mp4'  # Change this to your input video pathq

# Parameters
canny_threshold1 = 50
canny_threshold2 = 150
hough_threshold = 100
min_line_length = 100
max_line_gap = 10

cap = cv2.VideoCapture(VIDEO_PATH)


if not cap.isOpened():
    print(f"Error opening video file: {VIDEO_PATH}")
    exit(1)

while True:
    
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img,100,200,apertureSize = 3)
    if not ret:
        break
    B = frame[:,:,2]
    Y = 255-B
    thresh = cv2.adaptiveThreshold(Y,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        cv2.THRESH_BINARY_INV,35,5)
    
    contours, hierarchy = cv2.findContours(edges,  
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    # Hough Line Transform
    x=[]
    for i in range(0, len(contours)):
        if cv2.contourArea(contours[i]) > 100:
            x.append(contours[i])
    cv2.drawContours(img, x, -1, (255,0,0), 3) 

    cv2.imshow('Court Line Detection (press q or ESC to quit)', img)
    key = cv2.waitKey(20)
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows() 

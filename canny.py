import cv2
import numpy as np
import math

VIDEO_PATH = 'samples/point.mp4'  # Change this to your input video path

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

# Read only the first frame
ret, frame = cap.read()
if ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,200,apertureSize = 3)
    #lines = cv2.HoughLines(edges, 1, np.pi/180, 2, None, 30, 5)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 2, None, 30, 15)
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            print(l)
            cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, cv2.LINE_4)
    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #         cv2.line(frame, pt1, pt2, (255,0,0), 1, cv2.LINE_AA)
    cv2.imshow('dst',frame)
    cv2.imshow('Court Line Detection (press q or ESC to quit)', edges)
    # Wait for key press to close
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows() 

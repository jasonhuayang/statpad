import cv2
import numpy as np

# Define the video file to use for motion detection
video_file = "samples/test_clip.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_file)

# Get video properties for output video
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create video writer for foreground mask
output_mask_file = "fg_mask_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
mask_writer = cv2.VideoWriter(output_mask_file, fourcc, fps, (width, height), isColor=False)

# Create background subtractor (MOG2 is good for varying lighting)
background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

# Optional: Set minimum contour area to filter out noise
max_contour_area = 300
def preprocess_frame(frame):
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #frame = cv2.GaussianBlur(frame, (5, 5), 0)
    # frame = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    # frame = cv2.Canny(frame,100,200)
    # lower = np.array([25, 0, 0]) 
    # upper = np.array([45, 255, 255]) 
    # mask = cv2.inRange(frame, lower, upper)
    # frame = cv2.bitwise_and(frame,frame, mask=mask)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Typical tennis-ball yellow range
    lower_yellow = np.array([25, 60, 60])
    upper_yellow = np.array([40, 255, 255])
    mask_color = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Background subtraction
    fgmask = backsub.apply(frame, learningRate=0.002)

    # Combine color + motion
    combined = cv2.bitwise_and(mask_color, fgmask)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # Optional: blur to reduce noise
    combined = cv2.GaussianBlur(combined, (5,5), 0)
    return combined

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = preprocess_frame(frame)
    fg_mask = background_subtractor.apply(frame)
    
    # Apply morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours of moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around detected motion
    for contour in contours:
        if cv2.contourArea(contour) > max_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Motion", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the original frame with motion detection overlay
    cv2.imshow("Motion Detection", frame)
    
    # Optional: Display the foreground mask
    cv2.imshow("Foreground Mask", fg_mask)
    
    # Write the foreground mask frame to the output video
    mask_writer.write(fg_mask)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
mask_writer.release()
cv2.destroyAllWindows()


from ultralytics import YOLO
from collections import defaultdict

import cv2
import numpy as np

video_file = "samples/test_clip.mp4"  # replace with your video file path
model = YOLO('training/weights/best.pt')
cap = cv2.VideoCapture(video_file)

# Get video dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Array to store ball coordinates for each frame
ball_positions = []

# Define ROI size parameters
base_roi_size = 100  # base size of ROI
roi_size = base_roi_size
min_roi_size = 200   # minimum ROI size
max_roi_size = 800   # maximum ROI size
distance_scale = 2.0 # how much to scale ROI based on distance

# Define initial ROI position (will be updated with first detection)
roi_x = 640  # center x-coordinate
roi_y = 360  # center y-coordinate

# Find initial ball detection
print("Finding initial ball position...")
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Run detection on full frame
        results = model(frame, conf=0.9)
        if results[0].boxes:
            boxes = results[0].boxes.xywh.cpu()
            # Get first detection
            x, y = boxes[0][0], boxes[0][1]
            roi_x = int(x)
            roi_y = int(y)
            print(f"Initial ball position found at: ({roi_x}, {roi_y})")
            break
    else:
        print("No ball found in video")
        break

# Reset video capture to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Store last detection for distance calculation
last_detection = None
# Counter for consecutive missed detections
missed_detections = 0
# Growth rate for ROI when ball is not found
missed_growth_rate = 1.2

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Create a copy of the frame for visualization
        display_frame = frame.copy()
        
        # Ensure ROI coordinates stay within frame boundaries
        roi_x = max(roi_size//2, min(roi_x, frame_width - roi_size//2))
        roi_y = max(roi_size//2, min(roi_y, frame_height - roi_size//2))
        
        # Draw ROI rectangle on display frame
        cv2.rectangle(display_frame, 
                     (roi_x - roi_size//2, roi_y - roi_size//2),
                     (roi_x + roi_size//2, roi_y + roi_size//2),
                     (0, 255, 0), 2)
        
        # Extract ROI with boundary checks
        roi = frame[roi_y - roi_size//2:roi_y + roi_size//2,
                   roi_x - roi_size//2:roi_x + roi_size//2]
        print("running tracker")
        # Run YOLO tracking only on the ROI
        results = model.track(roi, conf=0.2, persist=True, tracker="ball_tracker.yaml")
        print("ran tracker")
        # Get the annotated ROI
        annotated_roi = results[0].plot()
        
        # Place the annotated ROI back into the display frame
        display_frame[roi_y - roi_size//2:roi_y + roi_size//2,
                     roi_x - roi_size//2:roi_x + roi_size//2] = annotated_roi

        # Store ball coordinates if detected
        frame_position = None
        if results[0].boxes:
            boxes = results[0].boxes.xywh.cpu()
            for box in boxes:
                x, y = box[0], box[1]
                # Convert ROI coordinates to global frame coordinates
                global_x = x + (roi_x - roi_size//2)
                global_y = y + (roi_y - roi_size//2)
                frame_position = (global_x, global_y)
                print("ball")

                # Calculate distance and scale ROI if we have a previous detection
                if last_detection is not None:
                    # Calculate Euclidean distance between current and last detection
                    distance = np.sqrt((global_x - last_detection[0])**2 + (global_y - last_detection[1])**2)
                    # Scale ROI size based on distance
                    roi_size = int(base_roi_size + distance * distance_scale)
                    # Ensure ROI size stays within bounds
                    roi_size = max(min_roi_size, min(roi_size, max_roi_size))
                    print(f"Distance: {distance:.1f}, New ROI size: {roi_size}")

                # Update ROI center to the detected ball position
                roi_x = int(global_x)
                roi_y = int(global_y)
                last_detection = (global_x, global_y)
                
                # Reset ROI size and missed detections counter when ball is found
                if missed_detections > 0:
                    roi_size = base_roi_size
                    print(f"Ball found! Resetting ROI size to {base_roi_size}")
                missed_detections = 0
                break  # Only take the first detection if multiple exist
        else:
            # Increment missed detections counter
            missed_detections += 1
            # Increase ROI size based on consecutive misses
            if missed_detections > 0:
                roi_size = min(max_roi_size, int(roi_size * missed_growth_rate))
                print(f"Missed detection {missed_detections}, increasing ROI size to {roi_size}")

        ball_positions.append(frame_position)

        # Display the frame with ROI
        cv2.imshow("YOLO11 Tracking", display_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

print(ball_positions)
print(f"Total frames processed: {len(ball_positions)}")
print(f"Frames with ball detection: {np.sum(ball_positions != None)}")

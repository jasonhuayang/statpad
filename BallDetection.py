from ultralytics import YOLO
from collections import defaultdict

import cv2
import numpy as np

video_file = "samples/test_clip.mp4"  # replace with your video file path
model = YOLO('training/weights/best.pt')
cap = cv2.VideoCapture(video_file)

# Array to store ball coordinates for each frame
ball_positions = []

# Define ROI size
roi_size = 250
# Define initial ROI position (you can adjust these coordinates as needed)
roi_x = 640  # center x-coordinate
roi_y = 360  # center y-coordinate

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Create a copy of the frame for visualization
        display_frame = frame.copy()
        
        # Draw ROI rectangle on display frame
        cv2.rectangle(display_frame, 
                     (roi_x - roi_size//2, roi_y - roi_size//2),
                     (roi_x + roi_size//2, roi_y + roi_size//2),
                     (0, 255, 0), 2)
        
        # Extract ROI
        roi = frame[roi_y - roi_size//2:roi_y + roi_size//2,
                   roi_x - roi_size//2:roi_x + roi_size//2]
        print("running tracker")
        # Run YOLO tracking only on the ROI
        results = model.track(roi, conf=0.3, persist=True, tracker="ball_tracker.yaml")
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
                # Update ROI center to the detected ball position
                roi_x = int(global_x)
                roi_y = int(global_y)
                break  # Only take the first detection if multiple exist
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

#ball_positions = np.array(ball_positions)
print(ball_positions)
print(f"Total frames processed: {len(ball_positions)}")
print(f"Frames with ball detection: {np.sum(ball_positions != None)}")

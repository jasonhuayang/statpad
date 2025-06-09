from ultralytics import YOLO
import cv2
import numpy as np

video_file = "samples/test_clip.mp4"  # replace with your video file path
model = YOLO('training/weights/best.pt')
cap = cv2.VideoCapture(video_file)

# Define ROI size
roi_size = 400
# Define ROI position (you can adjust these coordinates as needed)
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
        
        # Run YOLO tracking only on the ROI
        results = model.track(roi, conf=0.2, persist=True, tracker="ball_tracker.yaml")
        
        # Get the annotated ROI
        annotated_roi = results[0].plot()
        
        # Place the annotated ROI back into the display frame
        display_frame[roi_y - roi_size//2:roi_y + roi_size//2,
                     roi_x - roi_size//2:roi_x + roi_size//2] = annotated_roi

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

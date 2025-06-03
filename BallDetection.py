from ultralytics import YOLO
import cv2

video_file = "bosse.mp4"  # replace with your video file path
cap = cv2.VideoCapture(video_file)
model = YOLO('training/weights/best.pt')
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

#result = model.track(video_file, conf=0.4, persist=True, save=True, tracker="bytetrack.yaml")

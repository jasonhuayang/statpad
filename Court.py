from inference import get_model
import supervision as sv
import cv2

# define the video file to use for inference
video_file = "samples/point2.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_file)

# load a pre-trained yolov8n model
model = get_model(model_id="court-qo45v-lpwgg/1", api_key="Iuzy6U3O9RSmPobUEquD")

# create supervision annotators

edge_annotator = sv.EdgeAnnotator()

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    key_points = sv.KeyPoints(frame)
    edge_annotator = sv.EdgeAnnotator(
        color=sv.Color.GREEN,
        thickness=5
    )
    annotated_frame = edge_annotator.annotate(
        scene=frame.copy(),
        key_points=key_points
    )
    # cv2.imshow("Processed Video", annotated_frame)
    sv.plot_image(annotated_frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
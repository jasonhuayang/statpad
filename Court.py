from inference import get_model
import supervision as sv
import cv2

# define the video file to use for inference
video_file = "samples/bosse.mp4"  # Replace with your video file path
# cap = cv2.VideoCapture(video_file)
frame_generator = sv.get_video_frames_generator(video_file)
cap = cv2.VideoCapture(video_file)
frame = next(frame_generator)
# load a pre-trained yolov8n model
model = get_model(model_id="court-qo45v-lpwgg/1", api_key="Iuzy6U3O9RSmPobUEquD")

vertex_annotator = sv.VertexAnnotator(
    color=sv.Color.GREEN,
    radius=8
)
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Create a copy of the frame for visualization
        display_frame = frame.copy()

        result = model.infer(frame)[0]
        annotated_frame = frame.copy()
        if result:
            key_points = sv.KeyPoints.from_inference(result)
            annotated_frame = vertex_annotator.annotate(annotated_frame, key_points)

        cv2.imshow("Processed Video", annotated_frame)
        #sv.plot_image(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        # Break the loop if the end of the video is reached
        break


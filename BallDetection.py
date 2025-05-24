from inference import get_model
import supervision as sv
import cv2

# define the video file to use for inference
video_file = "Sample.mp4"  # replace with your video file path
cap = cv2.VideoCapture(video_file)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create video writer for output
output_file = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# load a pre-trained yolov8n model
model = get_model(model_id="moving-pickleball/3", api_key="ilhpCy5yCMhZbEGSzaOV")

# create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # run inference on the frame
    results = model.infer(frame)[0]
    
    # load the results into the supervision Detections api
    detections = sv.Detections.from_inference(results)
    
    # annotate the frame with our inference results
    annotated_frame = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections)
    
    # write the annotated frame to output video
    out.write(annotated_frame)
    
    # display the frame (optional)
    cv2.imshow('Processed Frame', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
cap.release()
out.release()
cv2.destroyAllWindows()
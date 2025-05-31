from ultralytics import YOLO
import supervision as sv
import numpy as np

# Initialize the model and trackers
model = YOLO('training/weights/best.pt')
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    # Run YOLO detection
    results = model(frame)[0]  # Get first result
    
    # Convert YOLO results to supervision format
    detections = sv.Detections(
        xyxy=results.boxes.xyxy.cpu().numpy(),
        confidence=results.boxes.conf.cpu().numpy(),
        class_id=results.boxes.cls.cpu().numpy().astype(int)
    )
    
    # Update tracker with new detections
    detections = tracker.update_with_detections(detections)
    
    # Create labels with class names and tracker IDs
    labels = [
        f"#{tracker_id} {model.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]
    
    # Annotate the frame with boxes and labels
    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
    return label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)

# Process the video with tracking
sv.process_video(
    source_path="samples/bosse.mp4",
    target_path="result.mp4",
    callback=callback
)
from ultralytics import YOLO
import cv2

video_file = "test_clip.mp4"  # replace with your video file path
model = YOLO('training/weights/best.pt')
result = model.track(video_file, conf=0.2, persist=True, save=True, tracker="ball_tracker.yaml")
print(result)
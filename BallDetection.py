from ultralytics import YOLO

video_file = "Sample.mp4"  # replace with your video file path
model = YOLO('training/weights/best.pt')
result = model.track(video_file, conf=0.2, save=True)
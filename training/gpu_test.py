import torch
import torch_directml
from ultralytics import YOLO
import os

# Disable CUDA checks
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Initialize DirectML device
device = torch_directml.device()
print(f"Using device: {device}")

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")  # load a pretrained model

    # Train the model with DirectML device
    results = model.train(
        data="training/data.yaml",
        epochs=100,
        imgsz=640,
        device=device,  # Use DirectML device
        batch=8,  # Reduced batch size to prevent memory issues
        amp=False,  # Disable automatic mixed precision
        workers=0  # <--- Add this to avoid multiprocessing issues on Windows
    )
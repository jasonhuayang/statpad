from inference import get_model
import supervision as sv
import cv2

# define the video file to use for inference
video_file = "samples/point2.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_file)
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed_frame = cv2.Canny(gray, 50, 150)
    processed_frame = cv2.GaussianBlur(processed_frame,(5,5),0)
    return processed_frame

# Read the first frame
ret, frame = cap.read()
if ret:
    processed_frame = preprocess_frame(frame)
    
    # Save the processed frame to disk
    output_filename = "canny_first_frame_blurred.png"
    cv2.imwrite(output_filename, processed_frame)
    
    sv.plot_image(processed_frame)
else:
    print("Error: Could not read the first frame from the video file")

# Release resources
cap.release()
cv2.destroyAllWindows()
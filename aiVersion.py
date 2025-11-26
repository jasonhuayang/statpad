import cv2
import numpy as np

# -------------------------
# 1. Initialize components
# -------------------------

# Background subtractor
backsub = cv2.createBackgroundSubtractorKNN(
    history=500,
    dist2Threshold=400.0,
    detectShadows=False
)

# Kalman filter (constant velocity model)
def create_kalman():
    kf = cv2.KalmanFilter(4, 2)
    # state = [x, y, vx, vy]
    kf.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]], np.float32)
    kf.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf

kf = create_kalman()
measurement = np.zeros((2,1), np.float32)

# ---------------------------------------
# 2. Helper — detect ball in a frame
# ---------------------------------------

def detect_ball(frame):
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Typical tennis-ball yellow range
    lower_yellow = np.array([25, 60, 60])
    upper_yellow = np.array([40, 255, 255])
    mask_color = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Background subtraction
    fgmask = backsub.apply(frame, learningRate=0.002)

    # Combine color + motion
    combined = cv2.bitwise_and(mask_color, fgmask)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # Optional: blur to reduce noise
    combined = cv2.GaussianBlur(combined, (5,5), 0)

    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    # Filter contours by expected size
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20 < area < 2000:  # tune based on resolution
            (x,y), radius = cv2.minEnclosingCircle(cnt)
            if area > best_area:
                best_area = area
                best = (int(x), int(y), int(radius))

    return best, combined

# ---------------------------------------
# 3. Main tracking loop
# ---------------------------------------

cap = cv2.VideoCapture("samples/sample.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect ball
    result, mask = detect_ball(frame)

    # Kalman prediction
    prediction = kf.predict()
    pred_x, pred_y = int(prediction[0]), int(prediction[1])

    if result is not None:
        x, y, r = result

        # Measurement update
        measurement[0] = x
        measurement[1] = y
        kf.correct(measurement)

        # Draw detection
        cv2.circle(frame, (x,y), r, (0,255,0), 2)
        cv2.circle(frame, (x,y), 3, (0,255,0), -1)

    # Draw Kalman prediction (useful when ball is lost)
    cv2.circle(frame, (pred_x, pred_y), 5, (0,0,255), -1)

    # Visual debug
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import os
import numpy as np

def verify_annotations():
    image_dir = 'training/court/train/images'
    label_dir = 'training/court/train/labels'

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    num_images = len(image_files)
    current_index = 0

    while True:
        image_name = image_files[current_index]
        image_path = os.path.join(image_dir, image_name)
        
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(label_dir, label_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            key = cv2.waitKey(0)
            if key == ord('q') or key == 27: # 'q' or ESC
                break
            elif key == ord('n'):
                current_index = (current_index + 1) % num_images
            elif key == ord('p'):
                current_index = (current_index - 1 + num_images) % num_images
            continue

        height, width, _ = image.shape
        
        display_image = image.copy()

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    keypoints = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
                    
                    for i, (x, y) in enumerate(keypoints):
                        if x > 0 and y > 0: # Draw only visible keypoints
                            px, py = int(x * width), int(y * height)
                            cv2.circle(display_image, (px, py), 5, (0, 255, 0), -1)
                            cv2.putText(display_image, str(i), (px + 5, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        window_title = f'Image {current_index + 1}/{num_images}: {image_name} | (n)ext, (p)rev, (q)uit'
        cv2.imshow(window_title, display_image)

        key = cv2.waitKey(0)

        if key == ord('q') or key == 27: # 'q' or ESC
            break
        elif key == ord('n'):
            current_index = (current_index + 1) % num_images
            cv2.destroyAllWindows()
        elif key == ord('p'):
            current_index = (current_index - 1 + num_images) % num_images
            cv2.destroyAllWindows()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    verify_annotations() 
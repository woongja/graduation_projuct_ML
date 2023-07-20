import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

#이미지 크기 조정
def resize_image(image, size):
    resized_image = image.resize(size)
    return resized_image

# 이미지 정규화
def normalize_image(image):
    normalized_image = np.array(image) / 255.0
    return normalized_image

#데이터셋 전처리
def preprocess_dataset(dataset_folder, image_size):
    preprocessed_data = []
    
    for filename in os.listdir(dataset_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            mage_path = os.path.join(dataset_folder, filename)
            image = Image.open(image_path)
            #이미지 크기 조정
            resized_image = resize_image(image, image_size)
            # 이미지 정규화
            normalize_image = normalize_image(resized_image)
            # 전저리된 데이터 저장
            preprocessed_data.append(normalized_image)
            
    return preprocessed_data
            
dataset_folder = '/Users/woongjae/Desktop/gradu/project/train/images'

image_size = (244,244)

preprocess_dataset = preprocess_dataset(dataset_folder,image_size)
        
# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# 동영상 파일 사용시
# video_path = "path/to/your/video/file.mp4"
# cap = cv2.VideoCapture(video_path)

# webcam 사용시
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("f"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
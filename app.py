
# with landmarks drawing

# pip install opencv-python opencv-python-headless torch torchvision matplotlib timm mediapipe

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import mediapipe as mp
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Load MiDaS model for depth estimation
midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_model.eval()
midas_transform = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to apply a colored mask
def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                 image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                 image[:, :, c])
    return image

# Function to estimate depth
def estimate_depth(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = midas_transform(img).unsqueeze(0)
    with torch.no_grad():
        prediction = midas_model(img)
    prediction = F.interpolate(
        prediction.unsqueeze(1),
        size=frame.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    depth = prediction.cpu().numpy()
    return depth

# Function to draw landmarks
def draw_landmarks(image, landmarks, bbox, visibility_threshold=0.5):
    for landmark in landmarks:
        if landmark.visibility < visibility_threshold:
            continue
        x = int(landmark.x * (bbox[2] - bbox[0]) + bbox[0])
        y = int(landmark.y * (bbox[3] - bbox[1]) + bbox[1])
        cv2.circle(image, (x, y), 5, (194, 247, 50), -1)

# Open video stream
cap = cv2.VideoCapture("video.mp4")  # Use 0 for webcam, or provide a video file path
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("record.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 inference
    results = model(frame, classes=0)  # class 0 is person
    
    # Process detections
    mask_image = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    
    if len(results) > 0:
        boxes = results[0].boxes
        for box in boxes:
            # Get box coordinates
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            
            mask_image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = 1
            person_image = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            
            # MediaPipe pose estimation
            person_rgb = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
            result = pose.process(person_rgb)
            if result.pose_landmarks:
                draw_landmarks(frame, result.pose_landmarks.landmark, xyxy)

    # Estimate depth
    depth = estimate_depth(frame)

    # Apply colored mask to the frame
    color = [0, 0, 255]  # Red color for mask
    frame = apply_mask(frame, mask_image, color)

    # Overlay depth map (optional)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
    frame = cv2.addWeighted(frame, 0.7, depth_colormap, 0.3, 0)

    video_writer.write(frame)

    # Display the resulting frame
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.imshow('Frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close display windows
cap.release()
video_writer.release()
cv2.destroyAllWindows()



# without landmarks drawing

# pip install opencv-python opencv-python-headless torch torchvision matplotlib timm

# import cv2
# import torch
# import numpy as np
# import torch.nn.functional as F
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# from PIL import Image
# from ultralytics import YOLO

# # Load YOLOv8 model
# model = YOLO('yolov8s.pt')

# # Load MiDaS model for depth estimation
# midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
# midas_model.eval()
# midas_transform = Compose([
#     Resize((256, 256)),
#     ToTensor(),
#     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Function to apply a colored mask
# def apply_mask(image, mask, color, alpha=0.5):
#     for c in range(3):
#         image[:, :, c] = np.where(mask == 1,
#                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
#                                  image[:, :, c])
#     return image

# # Function to estimate depth
# def estimate_depth(frame):
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(img)
#     img = midas_transform(img).unsqueeze(0)
#     with torch.no_grad():
#         prediction = midas_model(img)
#     prediction = F.interpolate(
#         prediction.unsqueeze(1),
#         size=frame.shape[:2],
#         mode="bicubic",
#         align_corners=False,
#     ).squeeze()
#     depth = prediction.cpu().numpy()
#     return depth

# # Open video stream
# cap = cv2.VideoCapture("video.mp4")  # Use 0 for webcam, or provide a video file path

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # YOLOv8 inference
#     results = model(frame, classes=0)  # class 0 is 'person'
    
#     # Estimate depth
#     depth = estimate_depth(frame)
    
#     # Process detections
#     mask_image = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    
#     if len(results) > 0:
#         # Get the first result (since we only processed one image)
#         result = results[0]
        
#         # Get boxes for person class
#         boxes = result.boxes
        
#         for box in boxes:
#             # Get the box coordinates
#             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
#             mask_image[y1:y2, x1:x2] = 1
    
#     # Apply colored mask to the frame
#     color = [0, 0, 255]  # Red color for mask
#     frame = apply_mask(frame, mask_image, color)
    
#     # Overlay depth map (optional)
#     depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
#     frame = cv2.addWeighted(frame, 0.7, depth_colormap, 0.3, 0)
    
#     # Display the resulting frame
#     cv2.imshow('Frame', frame)
    
#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture object and close display windows
# cap.release()
# cv2.destroyAllWindows()
import cv2
import os
import time
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def extract_hand_landmarks(frame):
    """
    Extract 21 hand landmarks (x, y, z) from frame
    Returns: numpy array of shape (63,) or None if no hand detected
    """
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        # Get first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract landmarks as flat array [x1, y1, z1, x2, y2, z2, ...]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks)
    
    return None

def save_frame_to_dataset(frame, word_name, save_path="dataset"):
    """
    Extract hand landmarks and save to dataset
    """
    if frame is None:
        return False
    
    # Create folder structure
    word_folder = os.path.join(save_path, word_name.strip().lower())
    os.makedirs(word_folder, exist_ok=True)
    
    try:
        # Extract landmarks
        landmarks = extract_hand_landmarks(frame)
        
        if landmarks is None:
            print("⚠️ No hand detected in frame")
            return False
        
        # Save landmarks as numpy file
        filename = os.path.join(word_folder, f"{time.time_ns()}.npy")
        np.save(filename, landmarks)
        
        return True
        
    except Exception as e:
        print(f"Error saving frame: {e}")
        return False

def visualize_hand_landmarks(frame):
    """
    Draw hand landmarks on frame for visualization
    Returns frame with landmarks drawn
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = hands.process(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                rgb_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
    
    return cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

def get_dataset_info(save_path="dataset"):
    """Get information about the current dataset"""
    if not os.path.exists(save_path):
        return {}
    
    dataset_info = {}
    classes = [d for d in os.listdir(save_path) 
               if os.path.isdir(os.path.join(save_path, d))]
    
    for class_name in classes:
        class_path = os.path.join(save_path, class_name)
        num_samples = len([f for f in os.listdir(class_path) 
                          if f.endswith('.npy')])
        dataset_info[class_name] = num_samples
    
    return dataset_info
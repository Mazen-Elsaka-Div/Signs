import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset"

def normalize_landmarks(landmarks):
    """
    Normalize landmarks to be scale and translation invariant
    """
    if landmarks is None:
        return None
        
    landmarks = landmarks.copy().reshape(21, 3)
    
    # Center at origin (subtract mean)
    landmarks = landmarks - landmarks.mean(axis=0)
    
    # Normalize scale (divide by max distance from origin)
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    if max_dist > 0:
        landmarks = landmarks / max_dist
    
    return landmarks.flatten()

def augment_landmarks(landmarks):
    """
    Apply random augmentations to hand landmarks
    - Random rotation around center
    - Random scale
    - Random noise
    """
    landmarks = landmarks.copy().reshape(21, 3)  # Reshape to (21, 3)
    
    # Random rotation (around z-axis)
    angle = np.random.uniform(-0.3, 0.3)  # radians
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    landmarks = landmarks @ rotation_matrix.T
    
    # Random scale (90% to 110%)
    scale = np.random.uniform(0.9, 1.1)
    landmarks *= scale
    
    # Random noise
    noise = np.random.normal(0, 0.01, landmarks.shape)
    landmarks += noise
    
    return landmarks.flatten()

def load_dataset(augment=False, augment_factor=2):
    """
    Load dataset of hand landmarks
    Returns: X (landmarks), y (labels), encoder, num_classes
    """
    X, y = [], []
    
    if not os.path.exists(DATASET_PATH):
        return None, None, None, 0
    
    classes = [d for d in os.listdir(DATASET_PATH) 
               if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    if not classes:
        return None, None, None, 0

    for label in classes:
        folder = os.path.join(DATASET_PATH, label)
        for file in os.listdir(folder):
            if not file.endswith('.npy'):
                continue
            
            try:
                # Load landmarks
                landmarks = np.load(os.path.join(folder, file))
                
                # Ensure correct shape (63,)
                if landmarks.shape[0] != 63:
                    continue
                
                # Normalize landmarks for consistency
                landmarks = normalize_landmarks(landmarks)
                
                # Add original
                X.append(landmarks)
                y.append(label)
                
                # Add augmented versions
                if augment:
                    for _ in range(augment_factor):
                        aug_landmarks = augment_landmarks(landmarks)
                        X.append(aug_landmarks)
                        y.append(label)
                        
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

    if not X:
        return None, None, None, 0
    
    X = np.array(X, dtype=np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    
    return X, np.array(y), encoder, len(encoder.classes_)

def load_dataset_with_validation(augment=True, val_split=0.2):
    """Load dataset and split into train/validation sets"""
    X, y, encoder, num_classes = load_dataset(augment=augment)
    
    if X is None:
        return None, None, None, None, None, 0
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42, stratify=y
    )
    
    return X_train, X_val, y_train, y_val, encoder, num_classes
import cv2
import numpy as np

def is_leaf_detected(img_path):
    img = cv2.imread(img_path)
    if img is None: return False
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define ranges for green, yellow, and brown (common plant colors)
    # Green
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Yellow/Brown/Disease colors
    lower_brown = np.array([10, 40, 40])
    upper_brown = np.array([30, 255, 255])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Calculate percentage
    green_ratio = cv2.countNonZero(mask_green) / (img.shape[0] * img.shape[1])
    brown_ratio = cv2.countNonZero(mask_brown) / (img.shape[0] * img.shape[1])
    
    print(f"Path: {img_path}")
    print(f"Green ratio: {green_ratio:.4f}, Brown ratio: {brown_ratio:.4f}")
    
    # A leaf should have some green, OR a very high amount of brown/yellow
    return green_ratio > 0.05 or (green_ratio > 0.01 and brown_ratio > 0.1)

is_leaf_detected('datasets/archive/Sugarcane__bacterial_blight/S_BLB (99).JPG')
is_leaf_detected('test_face.jpg')

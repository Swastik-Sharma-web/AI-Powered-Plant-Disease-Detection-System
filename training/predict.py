import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.h5"
DATASETS_DIR = BASE_DIR / "datasets"

# Global Variables
IMG_SIZE = (224, 224)
MODEL = None
CLASS_NAMES = None

# Dictionary to hold advice corresponding to standard classes
# (Expand this dictionary based on your actual dataset folders)
ADVICE_DB = {
    "Healthy": "Plant appears very healthy. Continue optimal watering and nutrient management practices.",
    "Early Blight": "Apply appropriate fungicide containing chlorothalonil or copper. Remove and destroy infected lower leaves. Ensure adequate spacing.",
    "Late Blight": "Apply protective fungicides immediately. Remove and destroy all infected plant parts. Do not compost infected material.",
    "Powdery Mildew": "Apply sulfur or potassium bicarbonate based fungicides. Ensure good air circulation and avoid overhead watering.",
    "Leaf Spot": "Remove infected leaves. Apply copper-based fungicide. Avoid wetting the foliage when watering."
}

def load_prediction_model():
    """Lazy load the model to speed up backend startup"""
    global MODEL
    if MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")
        print("Loading deep learning model into memory...")
        MODEL = load_model(str(MODEL_PATH))
    return MODEL

def load_class_names():
    """
    Dynamically load class names from the dataset directory structure.
    ImageDataGenerator sorts classes alphanumerically by default.
    """
    global CLASS_NAMES
    if CLASS_NAMES is None:
        # Assuming PlantVillage is the base training corpus for classes
        train_dir = DATASETS_DIR / "PlantVillage" / "plantvillage dataset" / "color"
        if train_dir.exists():
            # Get sorted subdirectories just like flow_from_directory does
            classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
            CLASS_NAMES = classes
        else:
            # Fallback for when datasets aren't present locally yet 
            # (matches standard mock names from previous step)
            CLASS_NAMES = ["Early Blight", "Healthy", "Late Blight", "Leaf Spot", "Powdery Mildew"]
            print("Warning: Training Dataset dir not found. Using default CLASS_NAMES list.")
    return CLASS_NAMES

def is_leaf_detected(image_file):
    """
    OpenCV HSV Color masking to verify if the uploaded image is actually a plant leaf.
    Checks for minimum thresholds of Green and Brown/Yellow.
    """
    try:
        # Save stream state
        current_pos = image_file.tell()
        image_file.seek(0)
        
        # Read to memory for cv2
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Restore stream for Keras
        image_file.seek(current_pos)
        
        if img is None:
            return False
            
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Green mask
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Brown/Yellow mask
        lower_brown = np.array([10, 40, 40])
        upper_brown = np.array([30, 255, 255])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Calculate percentage
        green_ratio = cv2.countNonZero(mask_green) / (img.shape[0] * img.shape[1])
        brown_ratio = cv2.countNonZero(mask_brown) / (img.shape[0] * img.shape[1])
        
        # A leaf should have some green, OR a very high amount of brown/yellow
        return green_ratio > 0.05 or (green_ratio > 0.01 and brown_ratio > 0.1)
    except Exception as e:
        print(f"Leaf check error: {e}")
        return True # Fallback to true if parsing fails so the AI can still try

def predict_disease(image_file):
    """
    Takes an uploaded image file (bytes), preprocesses it, 
    runs prediction, and formats the output.
    """
    try:
        # Load and verify model & classes
        model = load_prediction_model()
        classes = load_class_names()
        
        # 1. Load and preprocess image
        # image_file should be standard binary IO
        img = Image.open(image_file)
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        img = img.resize(IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array = img_array / 255.0 # Rescale 0-1
        
        # 2. Predict
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        predicted_class_name = classes[class_idx]
        
        # 3. Format Output Status and Advice
        is_healthy = "healthy" in predicted_class_name.lower()
        health_status = "Healthy" if is_healthy else "Diseased"
        
        # Friendly disease name (remove underscores etc if datasets used them)
        friendly_disease_name = predicted_class_name.replace("_", " ").title()
        if is_healthy:
            friendly_disease_name = "None"
            
        # OOD Check 1: Explicit Non-Leaf Rejection
        if not is_leaf_detected(image_file):
            return {
                "error": "No leaf detected.",
                "diseaseName": "Prediction Failed",
                "confidenceScore": 0.0,
                "confidencePercent": "0%",
                "status": "Invalid",
                "advice": "The system could not detect a leaf in this image. Please upload a clear photo of a plant leaf."
            }
            
        # OOD Check 2: Low Confidence Rejection Threshold
        # Since this model was heavily biased and fast-trained on specifically leaves, 
        # random desk objects usually score under 0.60
        if confidence < 0.60:
             return {
                "error": "Unrecognized object.",
                "diseaseName": "Unrecognized Object",
                "confidenceScore": round(confidence, 4),
                "confidencePercent": f"{int(confidence * 100)}%",
                "status": "Unknown",
                "advice": "The AI is not confident this is a plant leaf. Please ensure the image is a clear, focused picture of a single leaf."
            }
        
        # Find closest matching advice from DB, default to generic
        advice = ADVICE_DB.get(friendly_disease_name, 
            "Isolate the plant if possible. Bring a sample to a local agricultural extension for exact treatment protocols." 
            if not is_healthy else ADVICE_DB["Healthy"])

        # 4. Return formatted JSON dictionary
        return {
            "diseaseName": friendly_disease_name,
            "confidenceScore": round(confidence, 4),
            "confidencePercent": f"{int(confidence * 100)}%",
            "status": health_status,
            "advice": advice
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        return {
            "error": str(e),
            "status": "Error",
            "diseaseName": "Unknown",
            "confidenceScore": 0.0,
            "confidencePercent": "0%",
            "advice": "System failed to process the image."
        }

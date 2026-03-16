import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "datasets" / "PlantVillage" / "plantvillage dataset" / "color"
MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.h5"

# Hyperparameters (must match training)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def evaluate_model():
    print("--- Evaluating Trained Model ---")
    
    if not MODEL_PATH.exists():
        print(f"Error: Trained model not found at {MODEL_PATH}")
        print("Please run train_model.py first.")
        return
        
    if not TEST_DIR.exists():
        print(f"Error: Test dataset directory not found at {TEST_DIR}")
        return

    # 1. Load the model
    print("Loading model...")
    model = load_model(str(MODEL_PATH))
    
    # 2. Setup Test Data Generator (No Augmentation, Only Rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False # IMPORTANT: Must be False for correct evaluation metrics
    )
    
    if test_generator.samples == 0:
        print("No test images found.")
        return
        
    # 3. Predict on Test Set
    print("\nRunning predictions on test set (10 batches for fast evaluation)...")
    eval_steps = 10
    predictions = model.predict(test_generator, steps=eval_steps)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes[:len(y_pred)]
    
    # 4. Generate Classification Report
    print("\n--- Classification Report ---")
    class_labels = list(test_generator.class_indices.keys())
    
    unique_classes = np.unique(y_true)
    target_names = [class_labels[i] for i in unique_classes]
    
    report = classification_report(y_true, y_pred, labels=unique_classes, target_names=target_names)
    print(report)
    
    # 5. Generate Confusion Matrix
    print("--- Generating Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix on Test Set (plantdoc)')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    cm_path = BASE_DIR / "training" / "confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
if __name__ == "__main__":
    evaluate_model()

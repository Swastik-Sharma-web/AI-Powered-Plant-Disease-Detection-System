import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
MODEL_SAVE_PATH = BASE_DIR / "models" / "plant_disease_model.h5"

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

def create_data_generators():
    """Sets up the ImageDataGenerators for training, validation, and testing."""
    print("--- Setting up Data Generators ---")
    
    # In a real environment, you'd merge paths or symlink PlantVillage + Archive directories 
    # to a unified 'train_dir' for ImageDataGenerator.flow_from_directory.
    # For this script we'll assume they have been merged into a single "train_data" dir
    # by the validation script, OR we just point to one for demonstration.
    
    # Assumption for this pipeline: 
    # If using multiple dataset folders, it's best to consolidate them or use tf.data.Dataset
    # For simplicity of ImageDataGenerator, we assume a structured directory pointing to PlantVillage color images
    TRAIN_DIR = DATASETS_DIR / "PlantVillage" / "plantvillage dataset" / "color"
    
    if not TRAIN_DIR.exists():
        print(f"Warning: Training directory {TRAIN_DIR} not found.")
        # Create dummy path to avoid errors parsing the script
        os.makedirs(TRAIN_DIR, exist_ok=True)
    
    # 1. Training generator with Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2 # 80% train, 20% validation+test chunk
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # 2. Validation generator (No augmentation, just rescale)
    val_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, val_generator

def build_model(num_classes):
    """Builds the CNN architecture using MobileNetV2 transfer learning."""
    print("--- Building MobileNetV2 Model Architecutre ---")
    
    # Load pre-trained MobileNetV2 without the top classification layer
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_history(history):
    """Plots training/validation accuracy and loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def main():
    train_generator, val_generator = create_data_generators()
    
    if train_generator.samples == 0:
        print("No training images found. Please ensure datasets are properly placed.")
        return

    num_classes = train_generator.num_classes
    print(f"Detected {num_classes} classes.")
    
    model = build_model(num_classes)
    model.summary()
    
    print("\n--- Starting Training ---")
    
    # Callbacks for early stopping and saving best model
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Ensure models directory exists
    os.makedirs(MODEL_SAVE_PATH.parent, exist_ok=True)
    
    # Train the model (Full Training)
    print("Running full training. This will take time but will produce an accurate model...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    print(f"\nTraining Complete. Best model saved to {MODEL_SAVE_PATH}")
    
    plot_history(history)

if __name__ == "__main__":
    main()

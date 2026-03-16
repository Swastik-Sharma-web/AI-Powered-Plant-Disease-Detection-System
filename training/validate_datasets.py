import os
import zipfile
import hashlib
from PIL import Image
from pathlib import Path

# Important: Set Pillow to handle large and truncated images robustly
Image.MAX_IMAGE_PIXELS = None
ImageFile = Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS_DIR = (Path(__file__).resolve().parent.parent / "datasets").resolve()
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png'}
TARGET_DATASETS = ["PlantVillage", "plantdoc", "archive"]

def extract_archives():
    """Extracts any zip files in the datasets directory."""
    print("--- Checking for compressed archives ---")
    for item in DATASETS_DIR.iterdir():
        if item.is_file() and item.suffix.lower() == '.zip':
            extract_dir = DATASETS_DIR / item.stem
            print(f"Extracting {item.name} to {extract_dir}...")
            if not extract_dir.exists():
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(item, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            print(f"Successfully extracted {item.name}")

def compute_hash(file_path):
    """Computes MD5 hash to detect duplicate images."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def validate_and_clean_datasets():
    """Validates image integrity, removes duplicates, and standardizes formats."""
    print("\n--- Validating Datasets ---")
    
    total_images = 0
    removed_corrupted = 0
    removed_duplicates = 0
    converted_images = 0
    
    # Store hashes to detect duplicates
    seen_hashes = set()
    
    for dataset_name in TARGET_DATASETS:
        dataset_path = DATASETS_DIR / dataset_name
        if not dataset_path.exists() or not dataset_path.is_dir():
            print(f"Dataset {dataset_name} not found or is not a directory. Skipping...")
            continue
            
        print(f"\nProcessing {dataset_name}...")
        
        # Traverse categories inside the dataset
        for category_dir in dataset_path.iterdir():
            if not category_dir.is_dir():
                continue
                
            for root, _, files in os.walk(category_dir):
                for file in files:
                    file_path = Path(root) / file
                    
                    # 1. Skip non-image files
                    if file_path.suffix.lower() not in SUPPORTED_FORMATS and file_path.suffix.lower() not in {'.webp', '.bmp', '.tiff'}:
                        continue
                        
                    try:
                        # 2. Check for corruption
                        img = Image.open(file_path)
                        img.verify()  # verify() doesn't read the whole image, just headers
                        
                        # Re-open to actually load image data and do operations
                        img = Image.open(file_path)
                        
                        # 3. Convert unsupported formats to RGB JPG
                        if file_path.suffix.lower() not in SUPPORTED_FORMATS:
                            new_path = file_path.with_suffix('.jpg')
                            rgb_im = img.convert('RGB')
                            rgb_im.save(new_path, 'JPEG', quality=95)
                            file_path.unlink() # Delete old file
                            file_path = new_path
                            converted_images += 1
                            
                        # 4. Check for duplicates
                        file_hash = compute_hash(file_path)
                        if file_hash in seen_hashes:
                            file_path.unlink()
                            removed_duplicates += 1
                            continue
                            
                        seen_hashes.add(file_hash)
                        total_images += 1
                        
                    except Exception as e:
                        print(f"Removing corrupted/unreadable file: {file_path}")
                        try:
                            file_path.unlink()
                            removed_corrupted += 1
                        except:
                            pass

    print("\n--- Validation Report ---")
    print(f"Total Valid Images Retained: {total_images}")
    print(f"Corrupted Files Removed: {removed_corrupted}")
    print(f"Duplicate Images Removed: {removed_duplicates}")
    print(f"Images Converted to Standard Format: {converted_images}")

def generate_report():
    """Generates a summary report of the datasets."""
    print("\n--- Final Dataset Summary ---")
    
    # Assuming standard split based on prompt:
    # Train: PlantVillage, archive
    # Test: plantdoc
    train_datasets = ["PlantVillage", "archive"]
    test_datasets = ["plantdoc"]
    
    total_train = 0
    total_test = 0
    class_counts = {}
    
    for dataset_name in TARGET_DATASETS:
        dataset_path = DATASETS_DIR / dataset_name
        if not dataset_path.exists():
            continue
            
        for category_dir in dataset_path.iterdir():
            if not category_dir.is_dir():
                continue
                
            class_name = category_dir.name
            count = len(list(category_dir.glob("*.*")))
            
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += count
            
            if dataset_name in train_datasets:
                total_train += count
            elif dataset_name in test_datasets:
                total_test += count
                
    print(f"Total Classes Found: {len(class_counts)}")
    print(f"Total Training Images (PlantVillage + archive): {total_train}")
    print(f"Total Testing Images (plantdoc): {total_test}")
    
    print("\nImages per class:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f" - {class_name}: {count}")

if __name__ == "__main__":
    extract_archives()
    validate_and_clean_datasets()
    generate_report()

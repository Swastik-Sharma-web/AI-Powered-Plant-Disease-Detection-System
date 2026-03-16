import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = MobileNetV2(weights='imagenet')

def get_imagenet_preds(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return decode_predictions(preds, top=5)[0]

print("--- HUMAN FACE IMAGE ---")
leaf_path = 'test_face.jpg'
for p in get_imagenet_preds(leaf_path): print(p)


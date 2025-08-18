import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Load model
model = tf.keras.models.load_model("road_quality_model_transfer_512_mobilenet.keras")

# Parameters
IMG_SIZE = (512, 512)

# Load and preprocess image
img_path = "test/bad_3.png"
img = load_img(img_path, target_size=IMG_SIZE)  
img_array = img_to_array(img)                   
img_array = np.expand_dims(img_array, axis=0)   
img_array = img_array.astype("float32")         

# Predict
pred = model.predict(img_array)[0][0]  # single output
print("Raw prediction:", pred)

# If binary classification
if pred >= 0.5:
    print("✅ Road well painted")
else:
    print("❌ Road not well painted")

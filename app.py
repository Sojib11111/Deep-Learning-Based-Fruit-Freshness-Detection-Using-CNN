import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import cv2


# ---------------- CONFIG ----------------
INPUT_SIZE = (300, 300)
MODEL_WEIGHTS_PATH = "Fruit_EfficientNetB3_final.h5"
THRESHOLD = 0.75  # Unknown object threshold

CLASS_NAMES = [
    'apple_fresh', 'apple_rotten',
    'banana_fresh', 'banana_rotten',
    'mango_fresh', 'mango_rotten',
    'orange_fresh', 'orange_rotten',
    'tomato_fresh', 'tomato_rotten'
]
NUM_CLASSES = len(CLASS_NAMES)

# ---------------- MODEL BUILD ----------------
def build_model(input_shape, num_classes):
    base = EfficientNetB3(include_top=False, weights=None, input_shape=input_shape)
    base.trainable = True
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model

model = build_model((INPUT_SIZE[0], INPUT_SIZE[1], 3), NUM_CLASSES)
model.load_weights(MODEL_WEIGHTS_PATH, by_name=True, skip_mismatch=True)
print("✅ Model loaded with weights!")

# ---------------- PREDICTION FUNCTION ----------------
def predict_image(img):
    # Convert PIL/Gradio image to OpenCV format
    img = np.array(img)
    if img.shape[-1] == 4:  # RGBA to RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img_resized = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    img_array = np.expand_dims(img_resized, axis=0)
    processed = preprocess_input(img_array)

    preds = model.predict(processed)[0]
    idx = np.argmax(preds)
    confidence = preds[idx]

    top3_idx = np.argsort(preds)[::-1][:3]
    top3_dict = {CLASS_NAMES[i]: f"{preds[i]*100:.2f}%" for i in top3_idx}

    # Unknown object detection
    if confidence < THRESHOLD:
        most_likely_class = CLASS_NAMES[idx]
        return {
            "Prediction": "Unknown Object",
            "Most Likely Match": most_likely_class,
            "Confidence (%)": f"{confidence*100:.2f}%",
            "Top-3 Predictions": top3_dict
        }

    # Normal prediction
    return {
        "Prediction": CLASS_NAMES[idx],
        "Confidence (%)": f"{confidence*100:.2f}%",
        "Top-3 Predictions": top3_dict
    }

# ---------------- GRADIO INTERFACE ----------------
title = "🍎 Fruit Freshness Classification"
description = """
Upload any fruit image (.jpg, .jpeg, .png, .bmp, .webp, .avif).  
The model detects Fresh vs Rotten fruits. Unknown objects will also be detected and the most likely match will be shown.
"""

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.JSON(),
    title=title,
    description=description
)

if __name__ == "__main__":
    demo.launch()

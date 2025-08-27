import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys

# -------------------------
# Configuración
MODEL_PATH = "/Users/cefuentes/Documents/memoria/Trabajo-de-Titulo/backend/media/models_trained/nombre 12S_cnn_model.h5"
LABELS = ["bike", "climbing", "descending","gymbike", "jumping", "running", "standing", "treadmill", "walking"]  
# -------------------------

def classify_sample(sample_path):
    model = load_model(MODEL_PATH)
    
    # Cargar muestra
    sample = np.load(sample_path)
    if sample.ndim == 3:
        sample = sample[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)
    elif sample.ndim == 4:
        sample = sample[np.newaxis, ...]  # (1, H, W, D, 1)∑

    # Predicción
    sample = sample.astype(np.float32)
    sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample) + 1e-8)

    predictions = model.predict(sample)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_label = LABELS[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]

    print(f"✅ Predicción: {predicted_label} (confianza: {confidence:.4f})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python classify_sample.py <ruta_sample.npy>")
    else:
        classify_sample(sys.argv[1])

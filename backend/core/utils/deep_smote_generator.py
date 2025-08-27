import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.signal import spectrogram
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from django.conf import settings
from .deep_smote_ae import train_autoencoder, apply_deep_smote, train_autoencoder_temporal, apply_deep_smote_temporal
from .metrics import basic_stats
from skimage.transform import resize
import torch

# Directorios
BASE_DATA_DIR = os.path.join(settings.BASE_DIR, 'media', 'Smartphone_Dataset')
OUTPUT_DIR = os.path.join(settings.BASE_DIR, 'media', 'generated_tensors')
MODEL_DIR = os.path.join(settings.BASE_DIR, 'media', 'models_trained')
os.makedirs(MODEL_DIR, exist_ok=True)

# Campos esperados
SENSOR_CHANNELS = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'mag_x', 'mag_y', 'mag_z']

# Parámetros del espectrograma
SPEC_FS = 50
SPEC_NPERSEG = 64
SPEC_NOVERLAP = 32
SPEC_SHAPE = (64, 64)

def compute_spec(signal, fs=SPEC_FS, nperseg=SPEC_NPERSEG, noverlap=SPEC_NOVERLAP):
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx = resize(Sxx, SPEC_SHAPE, mode='reflect', anti_aliasing=True).astype(np.float32)
    return Sxx

def load_saved_tensors_from_dir(base_dir):
    """
    Carga tensores preprocesados desde carpetas por label.
    Retorna:
        X: np.ndarray de tensores
        y: np.ndarray de etiquetas
    """
    X_list, y_list = [], []
    for label in os.listdir(base_dir):
        label_dir = os.path.join(base_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for file in os.listdir(label_dir):
            if not file.endswith('.npy'):
                continue
            file_path = os.path.join(label_dir, file)
            X = np.load(file_path)
            X_list.append(X)
            y_list.append(label)
    return np.array(X_list), np.array(y_list)


def save_tensors(X, y, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for label in np.unique(y):
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        idxs = np.where(y == label)[0]
        for i, idx in enumerate(idxs):
            filepath = os.path.join(label_dir, f"{label}_{i}.npy")
            np.save(filepath, X[idx])


def generate_synthetic_tensor_dataset(name: str, sample_count: int = 1000):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # -------------------- CNN --------------------
    CNN_REAL_DIR = os.path.join(settings.BASE_DIR, 'media', 'tensor_cnn_real')
    X_real_tensor, y_real_tensor = load_saved_tensors_from_dir(CNN_REAL_DIR)

    if len(X_real_tensor) == 0:
        raise ValueError("❌ No se encontraron tensores CNN preprocesados.")

    model_cnn = train_autoencoder(X_real_tensor, num_epochs=50)
    model_output_path_cnn = os.path.join(MODEL_DIR, f"{name}_SENSOR_CNN_AE_{timestamp}.pt")
    torch.save(model_cnn.state_dict(), model_output_path_cnn)

    X_synthetic_cnn, y_synthetic_cnn = apply_deep_smote(
        model_cnn, 
        X_real_tensor, 
        y_real_tensor, 
        sample_count=sample_count,
        filter_flat=True,            
        normalize_output=True       
    )
    output_path_cnn = os.path.join(OUTPUT_DIR, f"{name}_sensor_CNN_{timestamp}")
    save_tensors(X_synthetic_cnn, y_synthetic_cnn, output_path_cnn)

    real_output_path_cnn = os.path.join(OUTPUT_DIR, f"{name}_sensor_REAL_CNN_{timestamp}")
    save_tensors(X_real_tensor, y_real_tensor, real_output_path_cnn)

    # -------------------- LSTM --------------------
    LSTM_REAL_DIR = os.path.join(settings.BASE_DIR, 'media', 'tensor_lstm_real')
    X_real_seq, y_real_seq = load_saved_tensors_from_dir(LSTM_REAL_DIR)

    if len(X_real_seq) == 0:
        raise ValueError("❌ No se encontraron tensores LSTM preprocesados.")

    model_lstm = train_autoencoder_temporal(X_real_seq, num_epochs=50)
    model_output_path_lstm = os.path.join(MODEL_DIR, f"{name}_SENSOR_LSTM_AE_{timestamp}.pt")
    torch.save(model_lstm.state_dict(), model_output_path_lstm)

    X_synthetic_lstm, y_synthetic_lstm = apply_deep_smote_temporal(
        model_lstm, 
        X_real_seq, 
        y_real_seq, 
        sample_count=sample_count,
        filter_flat=True,
        normalize_output=True
    )
    output_path_lstm = os.path.join(OUTPUT_DIR, f"{name}_sensor_LSTM_{timestamp}")
    save_tensors(X_synthetic_lstm, y_synthetic_lstm, output_path_lstm)

    real_output_path_lstm = os.path.join(OUTPUT_DIR, f"{name}_sensor_LSTM_REAL_{timestamp}")
    save_tensors(X_real_seq, y_real_seq, real_output_path_lstm)


    return {
        "cnn": (output_path_cnn, len(X_synthetic_cnn)),
        "lstm": (output_path_lstm, len(X_synthetic_lstm))
    }

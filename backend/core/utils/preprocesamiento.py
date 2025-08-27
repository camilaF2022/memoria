import os
import re
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.signal import spectrogram
from skimage.transform import resize

# ------------ CONFIGURACIÓN FIJA ------------ #

BASE_DATA_DIR = '/Users/cefuentes/Documents/memoria/Trabajo-de-Titulo/backend/media/Smartphone_Dataset'
CNN_REAL_DIR = '/Users/cefuentes/Documents/memoria/Trabajo-de-Titulo/backend/media/tensor_cnn_real'
LSTM_REAL_DIR = '/Users/cefuentes/Documents/memoria/Trabajo-de-Titulo/backend/media/tensor_lstm_real'

os.makedirs(CNN_REAL_DIR, exist_ok=True)
os.makedirs(LSTM_REAL_DIR, exist_ok=True)

SENSOR_CHANNELS = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'mag_x', 'mag_y', 'mag_z']

SPEC_FS = 50
SPEC_NPERSEG = 64
SPEC_NOVERLAP = 32
SPEC_SHAPE = (64, 64)

# ------------ FUNCIONES ------------ #

def compute_spec(signal: np.ndarray,
                 fs: int = SPEC_FS,
                 nperseg: int = SPEC_NPERSEG,
                 noverlap: int = SPEC_NOVERLAP) -> np.ndarray:
    _, _, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return resize(Sxx, SPEC_SHAPE, mode="reflect", anti_aliasing=True).astype(np.float32)

def normalize_minmax_local(tensor: np.ndarray) -> np.ndarray:
    mn = tensor.min(axis=0, keepdims=True)
    mx = tensor.max(axis=0, keepdims=True)
    return (tensor - mn) / (mx - mn + 1e-8)

def normalize_minmax_local_spec(tensor: np.ndarray) -> np.ndarray:
    mn = tensor.min(axis=(0, 1), keepdims=True)
    mx = tensor.max(axis=(0, 1), keepdims=True)
    return (tensor - mn) / (mx - mn + 1e-8)

def load_spectrogram_tensors() -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for subject in os.listdir(BASE_DATA_DIR):
        spath = os.path.join(BASE_DATA_DIR, subject)
        if not os.path.isdir(spath):
            continue
        for fname in os.listdir(spath):
            if not fname.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(spath, fname), header=None)
            if df.shape[0] < SPEC_NPERSEG:
                continue
            spec = np.stack([compute_spec(df[i].values) for i in range(df.shape[1])], axis=-1)
            spec = normalize_minmax_local_spec(spec)
            X.append(spec)
            y.append(re.sub(r"\d+", "", fname.split(".")[0].lower()))
    return np.asarray(X, dtype=np.float32), np.asarray(y)

def load_sequence_tensors(timesteps: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for subject in os.listdir(BASE_DATA_DIR):
        spath = os.path.join(BASE_DATA_DIR, subject)
        if not os.path.isdir(spath):
            continue
        for fname in os.listdir(spath):
            if not fname.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(spath, fname), header=None)
            if df.shape[0] < timesteps:
                continue
            data = df.values.astype(np.float32)
            data = normalize_minmax_local(data)
            for s in range(0, data.shape[0] - timesteps + 1, stride):
                X.append(data[s:s+timesteps])
                y.append(re.sub(r"\d+", "", fname.split(".")[0].lower()))
    return np.asarray(X, dtype=np.float32), np.asarray(y)

def save_tensors(X: np.ndarray, y: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for label in np.unique(y):
        ldir = os.path.join(out_dir, label)
        os.makedirs(ldir, exist_ok=True)
        for idx, i in enumerate(np.where(y == label)[0]):
            np.save(os.path.join(ldir, f"{label}_{idx}.npy"), X[i])

def preprocess_and_save_real_tensors(timesteps: int = 128, stride: int = 64) -> None:
    print("🌀 Generando tensores CNN (espectrogramas)…")
    X_spec, y_spec = load_spectrogram_tensors()
    save_tensors(X_spec, y_spec, CNN_REAL_DIR)
    print(f"✅ Guardados {len(X_spec)} tensores CNN en {CNN_REAL_DIR}")

    print("📈 Generando tensores LSTM (secuencias)…")
    X_seq, y_seq = load_sequence_tensors(timesteps, stride)
    save_tensors(X_seq, y_seq, LSTM_REAL_DIR)
    print(f"✅ Guardados {len(X_seq)} tensores LSTM en {LSTM_REAL_DIR}")

if __name__ == "__main__":
    preprocess_and_save_real_tensors()

import os
import numpy as np
import librosa
from datetime import datetime

# === CONFIGURACIÓN GLOBAL ===
AUDIO_PATH = '/Users/cefuentes/Documents/memoria/Trabajo-de-Titulo/backend/media/Audio'
CNN_REAL_AUDIO_DIR = '/Users/cefuentes/Documents/memoria/Trabajo-de-Titulo/backend/media/tensor_cnn_audio_real'
LSTM_REAL_AUDIO_DIR = '/Users/cefuentes/Documents/memoria/Trabajo-de-Titulo/backend/media/tensor_lstm_audio_real'
os.makedirs(CNN_REAL_AUDIO_DIR, exist_ok=True)
os.makedirs(LSTM_REAL_AUDIO_DIR, exist_ok=True)

# === FUNCIONES ===

def compute_audio_tensor(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    y = y[int(len(y) / 4):]
    hop_length = 512

    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
    log_mel = np.log1p(mel)  # Escala logarítmica para simular percepción humana

    # Normalización LOCAL por archivo (min-max solo del log-mel)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-9)

    if log_mel.shape[1] < 64:
        return None

    log_mel = log_mel[:64, :64]  # Cortamos a 64x64
    return np.expand_dims(log_mel, axis=-1)



def load_audio_tensors():
    X, y = [], []
    for label_folder in os.listdir(AUDIO_PATH):
        folder_path = os.path.join(AUDIO_PATH, label_folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.endswith(".mp3"):
                continue
            fpath = os.path.join(folder_path, fname)
            tensor = compute_audio_tensor(fpath)
            if tensor is None:
                continue
            label = label_folder.strip().lower().replace(" ", "_")
            X.append(tensor)
            y.append(label)
    return np.array(X), np.array(y)

def segment_audio_array(y_audio, segment_length=10*2000):
    """
    Segmenta un array de audio en múltiples segmentos de segment_length.
    """
    segments = []
    total_length = len(y_audio)
    step = segment_length 
    for start in range(0, total_length - segment_length + 1, step):
        segment = y_audio[start:start+segment_length]
        segments.append(segment)
    return segments

def load_audio_sequences():
    X_seq, y_seq = [], []
    segment_length = 10 * 8000  
    max_length = 45 * 8000      

    for label_folder in os.listdir(AUDIO_PATH):
        folder_path = os.path.join(AUDIO_PATH, label_folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.endswith(".mp3"):
                continue
            fpath = os.path.join(folder_path, fname)
            y_audio, sr = librosa.load(fpath, sr=8000)  
            y_audio = y_audio[int(len(y_audio) / 4):]  # elimina el primer 25%

            if len(y_audio) < max_length:
                y_audio = np.pad(y_audio, (0, max_length - len(y_audio)), mode='reflect')
            else:
                y_audio = y_audio[:max_length]

            y_audio = y_audio / (np.max(np.abs(y_audio)) + 1e-9)

            segments = segment_audio_array(y_audio, segment_length=segment_length)
            label = label_folder.strip().lower().replace(" ", "_")
            for segment in segments:
                X_seq.append(np.expand_dims(segment, axis=-1))  # shape: [timesteps, 1]
                y_seq.append(label)

    return np.array(X_seq, dtype=np.float32), np.array(y_seq)

def save_tensors(X, y, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for label in np.unique(y):
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        idxs = np.where(y == label)[0]
        for i, idx in enumerate(idxs):
            filepath = os.path.join(label_dir, f"{label}_{i}.npy")
            np.save(filepath, X[idx])

def preprocess_and_save_real_audio_tensors():
    print("🚀 Iniciando preprocesamiento de tensores reales de audio...")

    X_real_tensor, y_real_tensor = load_audio_tensors()
    if len(X_real_tensor) == 0:
        print("❌ No se encontraron datos de audio válidos en media/Audio para CNN.")
    else:
        print(f"✅ Tensores CNN generados: {X_real_tensor.shape}")
        save_tensors(X_real_tensor, y_real_tensor, CNN_REAL_AUDIO_DIR)
        print(f"✅ Tensores CNN guardados en {CNN_REAL_AUDIO_DIR}")

    X_real_seq, y_real_seq = load_audio_sequences()
    if len(X_real_seq) == 0:
        print("❌ No se encontraron datos de audio válidos en media/Audio para LSTM.")
    else:
        print(f"✅ Tensores LSTM generados: {X_real_seq.shape}")
        save_tensors(X_real_seq, y_real_seq, LSTM_REAL_AUDIO_DIR)
        print(f"✅ Tensores LSTM guardados en {LSTM_REAL_AUDIO_DIR}")

    print("🎉 Preprocesamiento de tensores de audio completado correctamente.")

# === EJECUCIÓN DIRECTA ===
if __name__ == "__main__":
    preprocess_and_save_real_audio_tensors()

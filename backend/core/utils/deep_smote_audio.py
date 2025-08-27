import os
import numpy as np
import torch
from datetime import datetime
from django.conf import settings
from .deep_smote_ae import train_autoencoder, apply_deep_smote, train_autoencoder_temporal, apply_deep_smote_temporal
from .deep_smote_generator import load_saved_tensors_from_dir, save_tensors

# === RUTAS ===
CNN_REAL_AUDIO_DIR = os.path.join(settings.BASE_DIR, 'media', 'tensor_cnn_audio_real')
LSTM_REAL_AUDIO_DIR = os.path.join(settings.BASE_DIR, 'media', 'tensor_lstm_audio_real')
OUTPUT_DIR = os.path.join(settings.BASE_DIR, 'media', 'generated_tensors')
MODEL_DIR = os.path.join(settings.BASE_DIR, 'media', 'models_trained')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def generate_synthetic_audio_dataset(name: str, sample_count: int = 1000):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ===== CNN: Espectrogramas =====
    X_real, y_real = load_saved_tensors_from_dir(CNN_REAL_AUDIO_DIR)
    if len(X_real) == 0:
        raise ValueError(f"❌ No se encontraron tensores reales CNN en {CNN_REAL_AUDIO_DIR}. Ejecuta el preprocesamiento primero.")

    model_cnn = train_autoencoder(X_real, num_epochs=200)
    model_output_path_cnn = os.path.join(MODEL_DIR, f"{name}_AUDIO_CNN_AE_{timestamp}.pt")
    torch.save(model_cnn.state_dict(), model_output_path_cnn)

    X_synthetic_cnn, y_synthetic_cnn = apply_deep_smote(model_cnn, X_real, y_real, sample_count)
    output_path_cnn = os.path.join(OUTPUT_DIR, f"{name}_audio_CNN_{timestamp}")
    save_tensors(X_synthetic_cnn, y_synthetic_cnn, output_path_cnn)

    # ===== LSTM: Secuencias Temporales =====
    X_seq_real, y_seq_real = load_saved_tensors_from_dir(LSTM_REAL_AUDIO_DIR)
    if len(X_seq_real) == 0:
        raise ValueError(f"❌ No se encontraron tensores reales LSTM en {LSTM_REAL_AUDIO_DIR}. Ejecuta el preprocesamiento primero.")
    print("Shape:", X_seq_real.shape)
    print("Min:", X_seq_real.min(), "Max:", X_seq_real.max(), "Mean:", X_seq_real.mean(), "Std:", X_seq_real.std())

    import matplotlib.pyplot as plt
    plt.plot(X_seq_real[0].squeeze())
    plt.title("Ejemplo de segmento de audio antes de entrenamiento")
    plt.savefig("qa_segmento_audio_lstm.png")
    plt.close()

    model_lstm = train_autoencoder_temporal(X_seq_real, tipo='audio', num_epochs=0, latent_dim=256)
    model_output_path_lstm = os.path.join(MODEL_DIR, f"{name}_AUDIO_LSTM_AE_{timestamp}.pt")
    torch.save(model_lstm.state_dict(), model_output_path_lstm)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_lstm.eval()
    with torch.no_grad():
        recon, _ = model_lstm(torch.tensor(X_seq_real[:10], dtype=torch.float32).permute(0, 2, 1).to(device))
    recon = recon.permute(0, 2, 1).cpu().numpy()


    import matplotlib.pyplot as plt
    for i in range(3):
        plt.figure(figsize=(10, 2))
        plt.plot(X_seq_real[i].squeeze(), label="Original")
        plt.plot(recon[i].squeeze(), label="Reconstruido")
        plt.title(f"Ejemplo reconstrucción {i}")
        plt.legend()
        plt.show()

    # X_synthetic_lstm, y_synthetic_lstm = apply_deep_smote_temporal(model_lstm, X_seq_real, y_seq_real, sample_count, filter_flat=False)
    output_path_lstm = os.path.join(OUTPUT_DIR, f"{name}_audio_LSTM_{timestamp}")
    # save_tensors(X_synthetic_lstm, y_synthetic_lstm, output_path_lstm)

    print("✅ Generación de datos sintéticos de audio completada correctamente.")

    return {
        "cnn": (output_path_cnn, len(X_synthetic_cnn)),
       
    }

    #  "lstm": (output_path_lstm, len(X_synthetic_lstm))

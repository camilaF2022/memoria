import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from django.conf import settings
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
from sklearn.svm import SVC
import yaml

def train_cnn_or_lstm(
    experiment_name: str,
    data_type: str = "cnn",
    dataset_folder: str = None,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 0.001
):
    models_dir = os.path.join(settings.BASE_DIR, 'media', 'models_trained')
    os.makedirs(models_dir, exist_ok=True)
    if dataset_folder is None:
        raise ValueError("dataset_folder es requerido.")

    labels = sorted([
        d for d in os.listdir(dataset_folder)
        if os.path.isdir(os.path.join(dataset_folder, d))
    ])
    X, y = [], []

    for label in labels:
        label_folder = os.path.join(dataset_folder, label)
        for fname in os.listdir(label_folder):
            if fname.endswith('.npy'):
                file_path = os.path.join(label_folder, fname)
                X.append(np.load(file_path))
                y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array([str(lbl) for lbl in y])  

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    class_names = encoder.classes_.tolist()
    num_classes = len(np.unique(y_encoded))
    print("📂 Carpetas encontradas (clases):", labels)
    print("✅ Clases únicas (LabelEncoder):", class_names)
    print("🔢 Etiquetas codificadas (ejemplo):", y_encoded[:10])
    print("📊 Distribución por clase (conteo):")
    from collections import Counter
    print(Counter(y_encoded))

    label_path = os.path.join(models_dir, f"{experiment_name}_{data_type}_classes.json")
    with open(label_path, 'w') as f:
        json.dump(class_names, f)

    if data_type == "cnn":
        if X.ndim != 4:
            raise ValueError(f"Tensor shape inválido para CNN: {X.shape}")
    
        class ImprovedCNN(nn.Module):
            def __init__(self, in_channels, num_classes):
                super().__init__()

                self.features = nn.Sequential(
                    nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),  # 64x64 → 32x32
                    nn.Dropout(0.25),

                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2),  # 32x32 → 16x16
                    nn.Dropout(0.25),

                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))  # 128 features por muestra
                )

                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes)
                )

            def forward(self, x):
                x = x.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
                x = self.features(x)
                x = self.classifier(x)
                return x

        model = ImprovedCNN(in_channels=X.shape[-1], num_classes=num_classes)

    elif data_type == "lstm":
        print("🔍 Mostrando secuencia temporal de ejemplo:")

        plt.figure(figsize=(10, 3))
        if X.shape[2] == 1:  # para tensores tipo (N, T, 1)
            plt.plot(X[0].squeeze())
        else:  # para multicanal
            for i in range(X.shape[2]):
                plt.plot(X[0, :, i], label=f'Canal {i}')
            plt.legend()
        plt.title("Ejemplo de secuencia temporal")
        plt.tight_layout()
        plt.savefig("qa_lstm_sequence_preview.png")  # guarda el gráfico
        plt.close()
        print("✅ Imagen guardada como 'qa_lstm_sequence_preview.png'")

        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        elif X.ndim == 3 and X.shape[2] not in [1, 2, 9]:
            raise ValueError(f"❌ Tensor shape inválido para LSTM: {X.shape}")
        input_dim = X.shape[2]


        class RobustLSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.4):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                    bidirectional=True
                )
                self.norm = nn.LayerNorm(hidden_dim * 2)  # bidirectional → 2x hidden_dim

                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim * 2, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(128, num_classes)
                )

            def forward(self, x):
                # x shape: (batch, seq_len, features)
                out, _ = self.lstm(x)
                out = out.mean(dim=1)  # global average pooling temporal
                out = self.norm(out)
                return self.classifier(out)

        model = RobustLSTM(
        input_dim=input_dim,
        hidden_dim=64,         # más capacidad
        num_layers=1,           # más profundidad
        num_classes=num_classes,
        dropout=0.0
    )


    else:
        raise ValueError("data_type debe ser 'cnn' o 'lstm'.")
    print("🧱 Shape final de X:", X.shape)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    print("🧪 X_tensor shape:", X_tensor.shape)
    print("📈 X_tensor min:", X_tensor.min().item(), "max:", X_tensor.max().item())
    print("🔠 y_tensor unique:", torch.unique(y_tensor, return_counts=True))

    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    inputs, targets = next(iter(train_loader))
    print("🧩 Batch shape:", inputs.shape)
    print("🎯 Targets:", targets[:10].numpy())

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        print("📊 Probabilidades de muestra:", probs[0].numpy())
    print("📈 Predicción:", torch.argmax(probs[0]).item())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    model_path = os.path.join(models_dir, f"{experiment_name}_{data_type}_model.pt")
    torch.save(model.state_dict(), model_path)

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{data_type.upper()} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(models_dir, f"{experiment_name}_{data_type}_loss_curve.png"))
    plt.close()

    try:
        all_preds, all_targets = [], []
        model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.numpy())
                all_targets.extend(targets.numpy())

        cm = confusion_matrix(all_targets, all_preds)
        cm_list = cm.tolist()

        combined_data = {
            "confusion_matrix": cm_list,
            "class_names": class_names
        }

    except Exception as e:
        print(f"❌ Error al guardar matriz y nombres de clases: {e}")
        confusion_path = None
        class_names_path = None


    return model_path, accuracy, avg_val_loss, combined_data


def train_naive_bayes_classifier(experiment_name: str, dataset_folder: str, var_smoothing=1e-9, test_split=0.2):
    models_dir = os.path.join(settings.BASE_DIR, 'media', 'models_trained')
    os.makedirs(models_dir, exist_ok=True)

    labels = sorted([
        d for d in os.listdir(dataset_folder)
        if os.path.isdir(os.path.join(dataset_folder, d))
    ])
    X, y = [], []

    for label in labels:
        label_folder = os.path.join(dataset_folder, label)
        for fname in os.listdir(label_folder):
            if fname.endswith('.npy'):
                file_path = os.path.join(label_folder, fname)
                X.append(np.load(file_path))
                y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array([str(lbl) for lbl in y])

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    class_names = encoder.classes_.tolist()
    print("🧠 Clases:", class_names)

    # Guardar clases
    label_path = os.path.join(models_dir, f"{experiment_name}_naive_bayes_classes.json")
    with open(label_path, 'w') as f:
        json.dump(class_names, f)

    # Aplanar espectrogramas: (N, 64, 64, 1) → (N, 4096)
    X_flat = X.reshape((X.shape[0], -1))

    # Separar en train y test
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=test_split, random_state=42)

    # Entrenar modelo
    model = GaussianNB(var_smoothing=var_smoothing)
    model.fit(X_train, y_train)

    # Evaluación
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cm_list = cm.tolist()

    print(f"🎯 Accuracy: {accuracy:.4f}")

    # Guardar modelo
    model_path = os.path.join(models_dir, f"{experiment_name}_naive_bayes_model.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Modelo guardado en {model_path}")

    combined_data = {
        "confusion_matrix": cm_list,
        "class_names": class_names
    }

    return model_path, accuracy, combined_data


def train_svm_classifier(experiment_name: str, dataset_folder: str, test_split=0.2, kernel='rbf', C=1.0, gamma='scale'):
    models_dir = os.path.join(settings.BASE_DIR, 'media', 'models_trained')
    os.makedirs(models_dir, exist_ok=True)

    labels = sorted([
        d for d in os.listdir(dataset_folder)
        if os.path.isdir(os.path.join(dataset_folder, d))
    ])
    X, y = [], []

    for label in labels:
        label_folder = os.path.join(dataset_folder, label)
        for fname in os.listdir(label_folder):
            if fname.endswith('.npy'):
                file_path = os.path.join(label_folder, fname)
                X.append(np.load(file_path))
                y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array([str(lbl) for lbl in y])

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    class_names = encoder.classes_.tolist()

    # Guardar clases
    label_path = os.path.join(models_dir, f"{experiment_name}_svm_classes.json")
    with open(label_path, 'w') as f:
        json.dump(class_names, f)

    # Aplanar espectrogramas: (N, H, W, C) o (N, H, W) → (N, H*W)
    if X.ndim == 4:
        X = X.squeeze(-1)  # (N, H, W, 1) → (N, H, W)
    X_flat = X.reshape((X.shape[0], -1))  # (N, H, W) → (N, H*W)

    # Separar en train y test
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=test_split, random_state=42)

    # Entrenar modelo
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)

    # Evaluación
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cm_list = cm.tolist()

    print(f"🎯 Accuracy SVM: {accuracy:.4f}")

    # Guardar modelo
    model_path = os.path.join(models_dir, f"{experiment_name}_svm_model.pkl")
    joblib.dump(model, model_path)
    print(f"💾 SVM guardado en {model_path}")

    combined_data = {
        "confusion_matrix": cm_list,
        "class_names": class_names
    }

    return model_path, accuracy, combined_data
def load_svm_data(folder_path):
    X, y = [], []
    labels = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    for label in labels:
        label_folder = os.path.join(folder_path, label)
        for fname in os.listdir(label_folder):
            if fname.endswith('.npy'):
                file_path = os.path.join(label_folder, fname)
                X.append(np.load(file_path))
                y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    class_names = encoder.classes_.tolist()

    if X.ndim == 4 and X.shape[-1] == 1:
        X = X.squeeze(-1)  # (N, H, W, 1) → (N, H, W)
    X_flat = X.reshape((X.shape[0], -1))  # (N, H, W) → (N, H*W)

    return X_flat, y_encoded, class_names

def crear_archivo_data_yaml(dataset_folder, class_names):
    data = {
        'path': dataset_folder,
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    output_path = os.path.join(dataset_folder, 'data.yaml')
    with open(output_path, 'w') as f:
        yaml.dump(data, f)
    return output_path
from ultralytics import YOLO

def train_yolo_model(dataset_folder, class_names, experiment_name="yolo_video", model_size="yolov8n.pt", epochs=50, imgsz=640, lr0=0.01):
    # Crear YAML
    yaml_path = crear_archivo_data_yaml(dataset_folder, class_names)

    # Crear modelo
    model = YOLO(model_size)

    # Entrenar
    model.train(data=yaml_path, epochs=epochs, imgsz=imgsz, lr0=lr0)
    # Guardar modelo
    model_output_path = os.path.join(dataset_folder, f"{experiment_name}_yolo.pt")
    model.save(model_output_path)

    return model_output_path
# import os
# import sys
# import django
# import os
# import json
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader, random_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# from django.conf import settings

# import os
# import sys
# import django

# # 👉 Agrega la carpeta raíz del proyecto al path de Python
# PROJECT_ROOT = "/Users/cefuentes/Documents/memoria/Trabajo-de-Titulo/backend"
# sys.path.append(PROJECT_ROOT)

# # ⚙️ Configura el módulo settings
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")  # <-- asegúrate que sea correcto
# django.setup()

# # Ruta al directorio con los tensores CNN reales
# CNN_REAL_DIR = "/Users/cefuentes/Documents/memoria/Trabajo-de-Titulo/backend/media/generated_tensors/a1_sensor_CNN_20250717_011713"
# def train_cnn_or_lstm(
#     experiment_name: str,
#     data_type: str = "cnn",
#     dataset_folder: str = None,
#     epochs: int = 30,
#     batch_size: int = 16,
#     lr: float = 0.001
# ):
#     if dataset_folder is None:
#         raise ValueError("dataset_folder es requerido.")

#     labels = sorted([
#         d for d in os.listdir(dataset_folder)
#         if os.path.isdir(os.path.join(dataset_folder, d))
#     ])
#     X, y = [], []

#     for label in labels:
#         label_folder = os.path.join(dataset_folder, label)
#         for fname in os.listdir(label_folder):
#             if fname.endswith('.npy'):
#                 file_path = os.path.join(label_folder, fname)
#                 X.append(np.load(file_path))
#                 y.append(label)

#     X = np.array(X, dtype=np.float32)
#     y = np.array([str(lbl) for lbl in y])  

#     encoder = LabelEncoder()
#     y_encoded = encoder.fit_transform(y)
#     class_names = encoder.classes_.tolist()
#     num_classes = len(np.unique(y_encoded))


#     if data_type == "cnn":
#         if X.ndim != 4:
#             raise ValueError(f"Tensor shape inválido para CNN: {X.shape}")
#         mean = X.mean(axis=(0, 1, 2), keepdims=True)
#         std = X.std(axis=(0, 1, 2), keepdims=True) + 1e-8
#         X = (X - mean) / std

#         class ImprovedCNN(nn.Module):
#             def __init__(self, in_channels, num_classes):
#                 super().__init__()

#                 self.features = nn.Sequential(
#                     nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
#                     nn.BatchNorm2d(32),
#                     nn.ReLU(),
#                     nn.Conv2d(32, 32, kernel_size=3, padding=1),
#                     nn.BatchNorm2d(32),
#                     nn.ReLU(),
#                     nn.MaxPool2d(2),  # 64x64 → 32x32
#                     nn.Dropout(0.25),

#                     nn.Conv2d(32, 64, kernel_size=3, padding=1),
#                     nn.BatchNorm2d(64),
#                     nn.ReLU(),
#                     nn.Conv2d(64, 64, kernel_size=3, padding=1),
#                     nn.BatchNorm2d(64),
#                     nn.ReLU(),
#                     nn.MaxPool2d(2),  # 32x32 → 16x16
#                     nn.Dropout(0.25),

#                     nn.Conv2d(64, 128, kernel_size=3, padding=1),
#                     nn.BatchNorm2d(128),
#                     nn.ReLU(),
#                     nn.AdaptiveAvgPool2d((1, 1))  # 128 features por muestra
#                 )

#                 self.classifier = nn.Sequential(
#                     nn.Flatten(),
#                     nn.Linear(128, 128),
#                     nn.ReLU(),
#                     nn.Dropout(0.3),
#                     nn.Linear(128, num_classes)
#                 )

#             def forward(self, x):
#                 x = x.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
#                 x = self.features(x)
#                 x = self.classifier(x)
#                 return x

#         model = ImprovedCNN(in_channels=X.shape[-1], num_classes=num_classes)

#     elif data_type == "lstm":
#         if X.ndim == 2:
#             X = X[:, :, np.newaxis]
#         elif X.ndim == 3 and X.shape[2] not in [1, 2, 9]:
#             raise ValueError(f"❌ Tensor shape inválido para LSTM: {X.shape}")
#         mean = X.mean(axis=(0, 1), keepdims=True)
#         std = X.std(axis=(0, 1), keepdims=True) + 1e-8
#         X = (X - mean) / std
#         input_dim = X.shape[2]


#         class SimpleLSTM(nn.Module):
#             def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
#                 super().__init__()
#                 self.lstm = nn.LSTM(
#                     input_dim,
#                     hidden_dim,
#                     num_layers,
#                     batch_first=True,
#                     dropout=0.3,
#                     bidirectional=True
#                 )
#                 self.fc = nn.Sequential(
#                     nn.Linear(hidden_dim * 2, 128),
#                     nn.ReLU(),
#                     nn.Linear(128, num_classes)
#                 )
#             def forward(self, x):
#                 out, _ = self.lstm(x)
#                 out = out.mean(dim=1)
#                 out = self.fc(out)
#                 return out
#         model = SimpleLSTM(input_dim, hidden_dim=64, num_layers=2, num_classes=num_classes)

#     else:
#         raise ValueError("data_type debe ser 'cnn' o 'lstm'.")

#     X_tensor = torch.tensor(X, dtype=torch.float32)
#     y_tensor = torch.tensor(y_encoded, dtype=torch.long)

#     dataset = TensorDataset(X_tensor, y_tensor)
#     train_size = int(0.7 * len(dataset))
#     val_size = int(0.15 * len(dataset))
#     test_size = len(dataset) - train_size - val_size
#     train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=lr)
#     train_losses, val_losses = [], []

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for inputs, targets in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         avg_train_loss = total_loss / len(train_loader)

#         model.eval()
#         total_val_loss = 0
#         with torch.no_grad():
#             for inputs, targets in val_loader:
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 total_val_loss += loss.item()
#         avg_val_loss = total_val_loss / len(val_loader)

#         train_losses.append(avg_train_loss)
#         val_losses.append(avg_val_loss)
#         print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

#     correct, total = 0, 0
#     model.eval()
#     with torch.no_grad():
#         for inputs, targets in test_loader:
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             total += targets.size(0)
#             correct += (predicted == targets).sum().item()
#     accuracy = correct / total

#     models_dir = os.path.join(settings.BASE_DIR, 'media', 'models_trained')
#     os.makedirs(models_dir, exist_ok=True)
#     model_path = os.path.join(models_dir, f"{experiment_name}_{data_type}_model.pt")
#     torch.save(model.state_dict(), model_path)

#     plt.figure(figsize=(8, 4))
#     plt.plot(train_losses, label='Train Loss')
#     plt.plot(val_losses, label='Val Loss')
#     plt.title(f'{data_type.upper()} Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig(os.path.join(models_dir, f"{experiment_name}_{data_type}_loss_curve.png"))
#     plt.close()

#     try:
#         all_preds, all_targets = [], []
#         model.eval()
#         with torch.no_grad():
#             for inputs, targets in test_loader:
#                 outputs = model(inputs)
#                 _, predicted = torch.max(outputs, 1)
#                 all_preds.extend(predicted.numpy())
#                 all_targets.extend(targets.numpy())

#         cm = confusion_matrix(all_targets, all_preds)
#         cm_list = cm.tolist()

#         combined_data = {
#             "confusion_matrix": cm_list,
#             "class_names": class_names
#         }

#         # combined_path = os.path.join(models_dir, f"{experiment_name}_{data_type}_confusion_with_classes.json")
#         # with open(combined_path, 'w') as f:
#         #     json.dump(combined_data, f)


#         # confusion_path = combined_path
#         # class_names_path = combined_path  

#     except Exception as e:
#         print(f"❌ Error al guardar matriz y nombres de clases: {e}")
#         confusion_path = None
#         class_names_path = None


#     return model_path, accuracy, avg_val_loss, combined_data

# if __name__ == "__main__":
#     experiment_name = "prueba_cnn_script"
#     data_type = "cnn"
#     dataset_folder = CNN_REAL_DIR
#     epochs = 20
#     batch_size = 16
#     learning_rate = 0.001

#     print("🚀 Iniciando entrenamiento CNN...")
#     model_path, accuracy, val_loss, confusion_data = train_cnn_or_lstm(
#         experiment_name=experiment_name,
#         data_type=data_type,
#         dataset_folder=dataset_folder,
#         epochs=epochs,
#         batch_size=batch_size,
#         lr=learning_rate
#     )

#     print("\n✅ Entrenamiento finalizado")
#     print(f"📦 Modelo guardado en: {model_path}")
#     print(f"🎯 Accuracy: {accuracy:.4f}")
#     print(f"📉 Validation Loss: {val_loss:.4f}")
#     print("📊 Matriz de confusión:")
#     print(confusion_data["confusion_matrix"])
#     print("🏷️ Clases:")
#     print(confusion_data["class_names"])
import matplotlib.pyplot as plt
import numpy as np

real = np.load("/Users/cefuentes/Documents/memoria/Trabajo-de-Titulo/backend/media/tensor_lstm_audio_real/datos_anomalos/datos_anomalos_7.npy")
fake = np.load("/Users/cefuentes/Documents/memoria/Trabajo-de-Titulo/backend/media/generated_tensors/sadee_audio_LSTM_20250718_131643/datos_anomalos/datos_anomalos_20.npy")
# real = np.load("/Users/cefuentes/Documents/memoria/Trabajo-de-Titulo/backend/media/tensor_lstm_audio_real/datos_normales/datos_normales_2.npy")
# fake = np.load("/Users/cefuentes/Documents/memoria/Trabajo-de-Titulo/backend/media/generated_tensors/Datos de audio maximo_audio_LSTM_20250718_115653/datos_normales/datos_normales_17.npy")
plt.figure(figsize=(14, 4))

plt.subplot(1, 2, 1)
plt.plot(real.squeeze())
plt.title("🔵 Secuencia REAL")

plt.subplot(1, 2, 2)
plt.plot(fake.squeeze())
plt.title("🔴 Secuencia GENERADA (DeepSMOTE)")

plt.show()

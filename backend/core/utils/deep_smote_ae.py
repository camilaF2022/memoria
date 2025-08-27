import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from imblearn.over_sampling import SMOTE
import pytorch_msssim
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from django.conf import settings

class SpectrogramAutoencoder(nn.Module):
    def __init__(self, in_channels=1, input_shape=(64, 64), latent_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_shape)
            x3, _, _ = self.forward_encoder(dummy)
            self.flat_dim = x3.view(1, -1).shape[1]
            self.unflatten_shape = x3.shape[1:]

        self.fc_latent = nn.Linear(self.flat_dim, latent_dim)
        self.fc_unlatent = nn.Linear(latent_dim, self.flat_dim)

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, in_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward_encoder(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        return x3, x2, x1

    def encode(self, x):
        x3, _, _ = self.forward_encoder(x)
        h_flat = x3.view(x.size(0), -1)
        latent = self.fc_latent(h_flat)
        return latent

    def decode(self, z):
        h = self.fc_unlatent(z)
        h_unflat = h.view(z.size(0), *self.unflatten_shape)
        x = self.dec3(h_unflat)
        x = self.dec2(x)
        x = self.dec1(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class TemporalAutoencoder(nn.Module):
    def __init__(self, input_length=512, latent_dim=256, in_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_length)
            encoded = self.encoder(dummy)
            self.flat_dim = encoded.view(1, -1).shape[1]
            self.unflatten_shape = encoded.shape[1:]
        self.fc_latent = nn.Linear(self.flat_dim, latent_dim)
        self.fc_unlatent = nn.Linear(latent_dim, self.flat_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=8, stride=2, padding=3, output_padding=0),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=2, padding=3, output_padding=0),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=2, padding=3, output_padding=0),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, in_channels, kernel_size=8, stride=2, padding=3, output_padding=0),
            nn.Sigmoid()
        )


    def encode(self, x):
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        latent = self.fc_latent(x_flat)
        return latent

    def decode(self, z):
        h = self.fc_unlatent(z)
        h_unflat = h.view(z.size(0), *self.unflatten_shape)
        x = self.decoder(h_unflat)
        return x

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

class TemporalAutoencoderAudio(nn.Module):
    def __init__(self, input_length=512, latent_dim=256, in_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_length)
            encoded = self.encoder(dummy)
            self.flat_dim = encoded.view(1, -1).shape[1]
            self.unflatten_shape = encoded.shape[1:]

        self.fc_latent = nn.Linear(self.flat_dim, latent_dim)
        self.fc_unlatent = nn.Linear(latent_dim, self.flat_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 64, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, in_channels, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.Tanh()  # ya que tus datos están en [-1, 1]
        )

    def encode(self, x):
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        latent = self.fc_latent(x_flat)
        return latent

    def decode(self, z):
        h = self.fc_unlatent(z)
        h_unflat = h.view(z.size(0), *self.unflatten_shape)
        x = self.decoder(h_unflat)
        return x

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z



def loss_fn(recon, target):
    mse = nn.functional.mse_loss(recon, target)
    ssim_loss = 1 - pytorch_msssim.ssim(recon, target, data_range=1, size_average=True)
    return 0.3 * mse + 0.7 * ssim_loss

def loss_fn2(recon, target):
    mse_loss = nn.MSELoss()(recon, target)
    return mse_loss


def train_autoencoder(X, num_epochs=300, lr=1e-4, batch_size=16, patience=30, latent_dim=64, checkpoint_name="ae_best.pt"):
    in_channels = X.shape[-1]
    input_shape = X.shape[1:3]
    model = SpectrogramAutoencoder(in_channels=in_channels, input_shape=input_shape, latent_dim=latent_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best_loss = float('inf')
    patience_counter = 0
    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        for (batch,) in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(dataset)
        loss_history.append(epoch_loss)
        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_name)
            print(f"✅ Checkpoint guardado: {checkpoint_name} (loss: {epoch_loss:.6f})")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"🛑 Early stopping en epoch {epoch+1}")
            break

    # Guardar curva de pérdida
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Curva de pérdida Autoencoder')
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.close()
    print("✅ Entrenamiento finalizado.")

    return model

def train_autoencoder_temporal(X, tipo='sensor', num_epochs=20, lr=1e-4, batch_size=16, patience=10, latent_dim=128, checkpoint_name="ae_temporal_best.pt"):
    
    if X.ndim == 3:
        in_channels = X.shape[2]  
    else:
        in_channels = 1  
    if tipo == 'audio':
        model = TemporalAutoencoderAudio( input_length=X.shape[1],
        latent_dim=latent_dim,
        in_channels=in_channels)
    else:
        model = TemporalAutoencoder(
        input_length=X.shape[1],
        latent_dim=latent_dim,
        in_channels=in_channels
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    if tipo == 'audio':
        loss_fn_custom = loss_fn2  # usa MSE + SSIM para audio
    else:
        loss_fn_custom = nn.MSELoss()  # MSE para sensor
    best_loss = float('inf  ')
    patience_counter = 0
    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        for (batch,) in dataloader:
            try:
                optimizer.zero_grad()
                recon, _ = model(batch)

                # Ajuste de longitud para evitar el error
                min_length = min(recon.shape[-1], batch.shape[-1])
                recon = recon[..., :min_length]
                batch = batch[..., :min_length]

                loss = loss_fn_custom(recon, batch)
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(f"❌ Error en entrenamiento: {e}")
                import traceback
                traceback.print_exc()
                break

            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        loss_history.append(epoch_loss)
        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_name)
            print(f"✅ Checkpoint guardado: {checkpoint_name} (loss: {epoch_loss:.6f})")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"🛑 Early stopping en epoch {epoch+1}")
            break

    return model


def apply_deep_smote(model, X, y, sample_count=1000, filter_flat=True, normalize_output=True):
    import warnings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        embeddings = model.encode(X_tensor).cpu().numpy()

    # SMOTE por clase
    unique_labels, counts = np.unique(y, return_counts=True)
    n_classes = len(unique_labels)
    samples_per_class = max(sample_count // n_classes, 10)
    target_strategy = {label: max(count, samples_per_class) for label, count in zip(unique_labels, counts)}

    print(f"👉 Clases detectadas: {n_classes}")
    print(f"👉 Estrategia SMOTE: {target_strategy}")

    smote = SMOTE(sampling_strategy=target_strategy, random_state=42)
    embeddings_resampled, y_resampled = smote.fit_resample(embeddings, y)
    print(f"✅ Embeddings SMOTE generados: {embeddings_resampled.shape}")

    # Reconstrucción
    embeddings_tensor = torch.tensor(embeddings_resampled, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructions = model.decode(embeddings_tensor).cpu()

    reconstructions = reconstructions.permute(0, 2, 3, 1).numpy()

    if filter_flat:
        stds = np.std(reconstructions, axis=(1, 2, 3))
        mask = stds > 1e-4
        reconstructions = reconstructions[mask]
        y_resampled = y_resampled[mask]

    # ✂️ Reducción final a sample_count máximo
    if len(reconstructions) > sample_count:
        indices = np.random.choice(len(reconstructions), sample_count, replace=False)
        reconstructions = reconstructions[indices]
        y_resampled = y_resampled[indices]

    return reconstructions, y_resampled



def apply_deep_smote_temporal(model, X, y, sample_count=1000, filter_flat=True, normalize_output=True):
    import warnings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # Preparar datos como tensores [B, C, T]
    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1).to(device)

    with torch.no_grad():
        embeddings = model.encode(X_tensor).cpu().numpy()

    # SMOTE
    unique_labels, counts = np.unique(y, return_counts=True)
    n_classes = len(unique_labels)
    samples_per_class = max(sample_count // n_classes, 10)
    target_strategy = {label: max(count, samples_per_class) for label, count in zip(unique_labels, counts)}
    
    print(f"👉 Clases detectadas: {n_classes}")
    print(f"👉 Estrategia SMOTE: {target_strategy}")

    smote = SMOTE(sampling_strategy=target_strategy, random_state=42)
    embeddings_resampled, y_resampled = smote.fit_resample(embeddings, y)
    print(f"✅ Embeddings SMOTE generados: {embeddings_resampled.shape}")

    # Reconstruir secuencias desde embeddings
    embeddings_tensor = torch.tensor(embeddings_resampled, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructions = model.decode(embeddings_tensor).cpu()

    # Volver a [B, T, C]
    reconstructions = reconstructions.permute(0, 2, 1).numpy()

    # 🚨 Filtrado de secuencias planas
        # 🚨 Filtrado de secuencias planas
    if filter_flat:
        stds = np.std(reconstructions, axis=(1, 2))
        mask = stds > 1e-4
        num_removed = np.sum(~mask)
        reconstructions = reconstructions[mask]
        y_resampled = y_resampled[mask]
        print(f"⚠️ Secuencias planas eliminadas: {num_removed}")

    # ✂️ Reducción final a sample_count máximo
    if len(reconstructions) > sample_count:
        indices = np.random.choice(len(reconstructions), sample_count, replace=False)
        reconstructions = reconstructions[indices]
        y_resampled = y_resampled[indices]

    return reconstructions, y_resampled



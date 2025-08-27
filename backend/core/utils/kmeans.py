# utils/kmeans_clustering.py
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from django.conf import settings
import matplotlib.pyplot as plt
import json

def run_kmeans_clustering(dataset_folder, n_clusters=3, save_plot=False, experiment_name="default"):
    labels = sorted([
        d for d in os.listdir(dataset_folder)
        if os.path.isdir(os.path.join(dataset_folder, d))
    ])
    
    X = []
    y_true = []

    for label in labels:
        label_folder = os.path.join(dataset_folder, label)
        for fname in os.listdir(label_folder):
            if fname.endswith('.npy'):
                file_path = os.path.join(label_folder, fname)
                X.append(np.load(file_path))
                y_true.append(label)

    X = np.array(X, dtype=np.float32)
    X_flat = X.reshape((X.shape[0], -1))  # (N, 64*64)

    print("📊 Shape de datos planos para clustering:", X_flat.shape)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_flat)

    if save_plot:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_flat)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7)
        plt.title(f"KMeans Clustering (k={n_clusters})")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.colorbar(scatter, label='Cluster')
        
        out_dir = os.path.join(settings.BASE_DIR, 'media', 'kmeans_plots')
        os.makedirs(out_dir, exist_ok=True)
        plot_path = os.path.join(out_dir, f"{experiment_name}_kmeans.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"🖼️ Plot guardado en: {plot_path}")
        return cluster_labels, plot_path

    return cluster_labels, None

# utils/kmeans_clustering.py

def train_kmeans_model(experiment_name: str, dataset_folder: str, n_clusters: int = 3):
    print(f"⚙️ Ejecutando KMeans con {n_clusters} clusters...")

    cluster_labels, plot_path = run_kmeans_clustering(
        dataset_folder=dataset_folder,
        n_clusters=n_clusters,
        save_plot=True,
        experiment_name=experiment_name
    )

    combined_data = {
        "plot_path": plot_path,
        "n_clusters": n_clusters,
        "labels_detected": list(np.unique(cluster_labels).astype(int))
    }

    output_folder = os.path.join(settings.BASE_DIR, 'media', 'models_trained')
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, f"{experiment_name}_kmeans_combined.json")

    with open(output_path, 'w') as f:
        json.dump(convert_numpy_types(combined_data), f)

    print(f"📁 Resultados de clustering guardados en: {output_path}")
    return output_path, combined_data

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    else:
        return obj

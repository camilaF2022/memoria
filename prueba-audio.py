from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import numpy as np
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min() + 1e-9)
    X_scaled = X_std * (max - min) + min
    return X_scaled

def spectrogram_image(y, sr, hop_length):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
    mels = np.log(mels + 1e-9) # add small number to avoid log(0)\
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    #img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy
    return img

def segment_spectrogram_image(img, return_points=False):
    # Flatten the image
    points = img.reshape(-1, 1)
    
    # If return_points is True, return the flattened points
    if return_points:
        return points
    
    # Initialize DBSCAN
    dbscan = DBSCAN(eps=0.9, min_samples=500)
    
    # Fit DBSCAN to the data
    dbscan.fit(points)
    
    # Get cluster labels
    labels = dbscan.labels_
    
    return labels.reshape(img.shape), dbscan


def plot_results(img, obj):
    # Downsample image
    scale = 32
    origshape = img.shape
    img = img[0::scale, 0::scale]

    # Flatten the image
    points = img.reshape(-1, 1)

    # Get cluster labels
    labels = obj.labels_
    labels = labels.reshape(origshape)
    labels = labels[0::scale, 0::scale]
    labels = labels.reshape(-1, 1)
    
    # Calculate centroids
    centroids = []
    for label in np.unique(labels):
        centroid = np.mean(points[labels == label])
        centroids.append(centroid)
    centroid_indices = np.arange(len(centroids))
    print(centroid_indices)
    print(labels)
    print(points.ravel())
    
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(points)), points.ravel(), c=labels, cmap='viridis', s=10)
    plt.scatter(centroid_indices, centroids, c='red', marker='x', s=100)  # Plot centroids
    plt.title('Scatterplot with Clusters and Centroids')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.colorbar(label='Cluster Label')
    plt.show()

if __name__ == '__main__':
    
    mp3filepath = '/Users/cefuentes/Downloads/Datos/Audio/Datos normales/2024-05-03T09-25.mp3'

    x, sr = librosa.load(mp3filepath)
    x = x[int(len(x)/4):]
    hop_length = 1024
    image = spectrogram_image(x, sr=sr, hop_length=hop_length)
    segmented_image, dbscan = segment_spectrogram_image(image)

    kmeans = KMeans(n_clusters=4, random_state=0).fit(segmented_image.reshape(-1,1))
    klabels = kmeans.labels_.reshape(segmented_image.shape)

    
    #klabels = scale_minmax(klabels, 0, 255).astype(np.uint8)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #klabels = cv2.morphologyEx(klabels, cv2.MORPH_OPEN, kernel)

    plt.figure(figsize=(16, 9))

    # Plot original spectrogram
    plt.subplot(5, 1, 1)
    plt.imshow(image, aspect='auto', origin='lower')
    plt.colorbar(label='Intensity')
    plt.title('Original Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency Bin')

    # Plot feature space
    plt.subplot(5, 1, 2)
    plt.scatter(range(len(segmented_image.flatten())), segmented_image.flatten(), c=segmented_image.flatten(), cmap='viridis', marker='.')
    plt.colorbar(label='Cluster')
    plt.title('Feature Space')
    plt.xlabel('Point Index')
    plt.ylabel('Cluster')
    plt.ylim(-1, np.max(segmented_image.flatten()) + 1)  # Adjust ylim to include all clusters

    # Plot segmented spectrogram
    plt.subplot(5, 1, 3)
    plt.imshow(segmented_image, aspect='auto', origin='lower')
    plt.colorbar(label='Cluster')
    plt.title('Segmented Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency Bin')


    # Plot new feature space
    plt.subplot(5, 1, 4)
    plt.scatter(range(len(klabels.flatten())), klabels.flatten(), c=klabels.flatten(), cmap='viridis', marker='.')
    plt.colorbar(label='Cluster')
    plt.title('Feature Space + Kmeans')
    plt.xlabel('Point Index')
    plt.ylabel('Cluster')
    plt.ylim(-1, np.max(klabels.flatten()) + 1)  # Adjust ylim to include all clusters

    # Plot segmented spectrogram
    plt.subplot(5, 1, 5)
    plt.imshow(klabels, aspect='auto', origin='lower')
    plt.colorbar(label='Cluster')
    plt.title('DBSCAN + Kmeans')
    plt.xlabel('Time')
    plt.ylabel('Frequency Bin')

    plt.tight_layout()
    plt.show()
    
    plot_results(segmented_image, kmeans)

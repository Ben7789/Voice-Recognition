import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os


audio_folder = '/Users/rendvalor/downloads/test5'
features = []
labels = []


for filename in os.listdir(audio_folder):
    filepath = os.path.join(audio_folder, filename)
    signal, sample_rate = librosa.load(filepath, sr=None, mono=True)

    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=100)
    mfcc_mean = np.mean(mfcc, axis=1)

    features.append(mfcc_mean)
    labels.append(filename)


features = StandardScaler().fit_transform(features)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features)


color_map = {
    'Z': 'red',
    'Andrea': 'blue',
    'Jonathan': 'green',
    'EB': 'purple',
    'FB': 'orange',
    'K': 'cyan',
    'Other': 'black'
}


plt.figure(figsize=(10, 7))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Speaker Clustering in Voice Space')


for i, label in enumerate(labels):
    if label.startswith('Z'):
        color = 'red'
        label_key = 'Z'
    elif label.startswith('Andrea'):
        color = 'blue'
        label_key = 'Andrea'
    elif label.startswith('Jonathan'):
        color = 'green'
        label_key = 'Jonathan'
    elif label.startswith('EB'):
        color = 'purple'
        label_key = 'EB'
    elif label.startswith('FB'):
        color = 'orange'
        label_key = 'FB'
    elif label.startswith('K'):
        color = 'cyan'
        label_key = 'K'
    else:
        color = 'black'
        label_key = 'Other'
    
    
    label_points = {label: [] for label in color_map.keys()}
    plt.scatter(principal_components[i, 0], principal_components[i, 1], color=color)
    label_points[label_key].append(principal_components[i])


for key, color in color_map.items():
    indices = [i for i, label in enumerate(labels) if label.startswith(key)]
    if indices:
        centroid = np.mean(principal_components[indices], axis=0)
        plt.scatter(centroid[0], centroid[1], color=color, marker='X', s=200, label=f'Centroid {key}')


handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in color_map.values()]
legend_labels = color_map.keys()
plt.legend(handles, legend_labels, title="Label Key", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('/Users/rendvalor/downloads/figure1_with_centroids.png')

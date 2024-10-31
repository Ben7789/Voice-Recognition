import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os


audio_folder = '/Users/rendvalor/downloads/compressed'
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


plt.figure(figsize=(10, 7))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Speaker Clustering in Voice Space')
plt.show()

for i, label in enumerate(labels):
    color = 'red' if label.startswith('Voice') else 'blue'  # Color based on filename
    plt.scatter(principal_components[i, 0], principal_components[i, 1], color=color, label=label)
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import soundfile as sf
import noisereduce as nr


audio_folder = '/Users/rendvalor/downloads/test5'
features = []
labels = []


def clean(audio, sr, noise=0.01, silence=-40):
    clean_audio = audio.copy()
    # Removes silence from audio and creates mask for silent parts
    audio_db = librosa.amplitude_to_db(np.abs(clean_audio), ref=np.max)
    speech = audio_db > silence
    # Applies mask
    audio_final = clean_audio[speech]
    #Reduces background noise
    audio_final = nr.reduce_noise(audio_final, sr=sr)
    #Plots signal before and after cleaning
    #plt.figure()
    #plt.plot(audio)
    #plt.figure()
    #plt.plot(audio_final)
    
    return audio_final

def pca(mfcc):
    meaned = mfcc - np.mean(mfcc, axis=0)
    standardised = meaned/np.std(mfcc, axis=0)
    cov = np.cov(standardised.T)
    values, vectors = np.linalg.eig(cov)
    idx = values.argsort()[::-1]
    values = values[idx]
    vectors = vectors[:,idx]
    pca = np.dot(standardised, vectors[:,:2])
    return pca


## Check that cleaned audio sounds better and only includes speech. Not needed for final product
##extract audio of choice
#af, sr = clean(r'C:\Users\LBlan\OneDrive - The University of Nottingham\Voice Recognition IDP Project\Audio samples\EB4.wav')
#output = 'clean6.wav'
#sf.write(output, af, sr)


noise = []
sound2 = sound.copy()
# Set a length of the list to 10
for i in range(len(sound)):
    noise1 = random.uniform(-0.05, 0.05)
    sound2[i]= sound[i] + noise1
    noise.append(noise1)

plt.figure()
plt.plot(noise[:1000])
plt.figure()
plt.plot(sound)
plt.figure()
plt.plot(sound2)






for filename in os.listdir(audio_folder):
    filepath = os.path.join(audio_folder, filename)
    signal, sample_rate = librosa.load(filepath, sr=None, mono=True)
    
    #cleans the audio
    signal= clean(signal, sample_rate)

    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=100)
    mfcc_mean = np.mean(mfcc, axis=1)

    features.append(mfcc_mean)
    labels.append(filename)


features = StandardScaler().fit_transform(features)
pca = PCA(n_components=2)
pca = pca(features)
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

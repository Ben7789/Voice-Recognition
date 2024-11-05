import numpy as np
import scipy.fftpack
import scipy.signal
import os
import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import librosa
import soundfile as sf
import noisereduce as nr

def mel_scale(freq):
    return 2595 * np.log10(1 + freq / 700)

def inverse_mel_scale(mel):
    return 700 * (10**(mel / 2595) - 1)

def pre_emphasis(signal):
    return np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

def create_mel_filter_bank(sample_rate, n_fft, n_mels):
    mel_points = np.linspace(mel_scale(0), mel_scale(sample_rate / 2), n_mels + 2)
    hz_points = inverse_mel_scale(mel_points)
    bin_points = np.round((n_fft + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((n_mels, (n_fft // 2 + 1)))
    
    for i in range(1, n_mels + 1):
        filters[i - 1, bin_points[i - 1]:bin_points[i]] = np.linspace(0, 1, bin_points[i] - bin_points[i - 1])
        filters[i - 1, bin_points[i]:bin_points[i + 1]] = np.linspace(1, 0, bin_points[i + 1] - bin_points[i])
        
    return filters

def extract_mfcc(signal, sample_rate, n_mfcc, n_fft=2048, hop_length=512, n_mels=42):
    emphasized_signal = pre_emphasis(signal)
    framed_signal = librosa.util.frame(emphasized_signal, frame_length=n_fft, hop_length=hop_length).T
    window = scipy.signal.windows.hamming(n_fft)
    windowed_frames = framed_signal * window
    power_spectrum = (np.abs(np.fft.rfft(windowed_frames, n_fft)) ** 2) / n_fft
    mel_filter_bank = create_mel_filter_bank(sample_rate, n_fft, n_mels)
    mel_spectrogram = np.dot(power_spectrum, mel_filter_bank.T)
    log_mel_spectrogram = np.log(mel_spectrogram + 1e-10)
    mfcc = scipy.fftpack.dct(log_mel_spectrogram, axis=1, norm='ortho')[:, :n_mfcc]
    return np.mean(mfcc, axis=0)

def clean(audio, sr, noise=0.01, silence=-70):
    clean_audio = audio.copy()
    audio_db = librosa.amplitude_to_db(np.abs(clean_audio), ref=np.max)
    speech = audio_db > silence
    audio_final = clean_audio[speech]
    audio_final = nr.reduce_noise(clean_audio, sr=sr)
    #plt.figure()
    #plt.plot(audio)
    #plt.figure()
    #plt.plot(audio_final)
    return audio_final

def pca(features):
    meaned = features - np.mean(features, axis=0)
    standardised = meaned / np.std(features, axis=0)
    cov = np.cov(standardised.T)
    values, vectors = np.linalg.eig(cov)
    idx = values.argsort()[::-1]
    values = values[idx]
    vectors = vectors[:, idx]
    pca = np.dot(standardised, vectors[:, :2])
    return pca

audio_folder = '/Users/rendvalor/Pictures/voice 2'
features = []
labels = []

for filename in os.listdir(audio_folder):
    filepath = os.path.join(audio_folder, filename)
    signal, sample_rate = librosa.load(filepath, sr=None, mono=True)
    '''signal= clean(signal, sample_rate)
    for i in range(len(signal)):
        noise = np.random.uniform(-0.01,0.01)
        signal[i] += noise'''
    mfcc_mean = extract_mfcc(signal, sample_rate, n_mfcc=42)
    features.append(mfcc_mean)
    labels.append(filename)

test_file_path = '/Users/rendvalor/pictures/test/Z-196.wav'
test_signal, test_sample_rate = librosa.load(test_file_path, sr=None, mono=True)
test_mfcc_mean = extract_mfcc(test_signal, test_sample_rate, n_mfcc=42)

features.append(test_mfcc_mean)
labels.append("Test Sample")

scaler = StandardScaler()
features = scaler.fit_transform(features)
principal_components = pca(features)

test_principal_component = principal_components[-1]
principal_components = principal_components[:-1]

color_map = {
    'Z': 'red',
    'Andrea': 'blue',
    'Jonathan': 'green',
    'EB': 'purple',
    'FB': 'orange',
    'GB': 'cyan',
    'Jont': 'magenta',
    'LB': 'black'
}

plt.figure(figsize=(10, 7))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Speaker Clustering in Voice Space')

for i in range(len(labels) - 1):
    label = labels[i]
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
    elif label.startswith('GB'):
        color = 'cyan'
        label_key = 'GB'
    elif label.startswith('Jont'):
        color = 'magenta'
        label_key = 'Jont'
    elif label.startswith('LB'):
        color = 'black'
        label_key = 'LB'
    
    plt.scatter(principal_components[i, 0], principal_components[i, 1], color=color)

centroids = []

for key, color in color_map.items():
    indices = []
    for i in range(len(labels)):
        if labels[i].startswith(key):
            indices.append(i)
    
    centroid = np.mean(principal_components[indices], axis=0)
    var_x = np.var(principal_components[indices][:,0])
    var_y = np.var(principal_components[indices][:,1])
    var_rad = np.sqrt((var_x + var_y) / 2)
    plt.scatter(centroid[0], centroid[1], color=color, marker='X', s=200, label=f'Centroid {key}')
    centroids.append(centroid)
    
    for i in range (1,5):
        circle = plt.Circle(centroid, i*var_rad, color=color, fill=False, linestyle='--', label='Variance Circle')
        plt.gca().add_artist(circle)
    

distances = np.zeros(len(centroids))

for i in range(len(centroids)):
    centroid = centroids[i]
    squared_diffs = np.abs(centroid - test_principal_component) ** 2
    distances[i] = np.sqrt(np.sum(squared_diffs))

smallest_distance = np.argmin(distances)

if smallest_distance == 0:
    print('Speaker is Z')
elif smallest_distance == 1:
    print('Speaker is Andrea')
elif smallest_distance == 2:
    print('Speaker is Jonathan')
elif smallest_distance == 3:
    print('Speaker is EB')
elif smallest_distance == 4:
    print('Speaker is FB')
elif smallest_distance == 5:
    print('Speaker is GB')
elif smallest_distance == 6:
    print('Speaker is Jont')
elif smallest_distance == 7:
    print('Speaker is LB')
else:
    print('Not in plot')

plt.scatter(test_principal_component[0], test_principal_component[1], color='yellow', marker='o', s=100, label='Test Sample')
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in color_map.values()]
legend_labels = list(color_map.keys())
handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10))
legend_labels.append('Test Sample')
plt.legend(handles, legend_labels, title="Label Key", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
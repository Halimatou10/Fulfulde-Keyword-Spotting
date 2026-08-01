import librosa
import numpy as np

def extraire_mfcc(chemin_fichier, n_mfcc=40, sr=16000):
    signal, sr = librosa.load(chemin_fichier, sr=sr)
    mfcc   = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0)
    return features, signal, sr

def normaliser(features):
    mean = np.mean(features, axis=1, keepdims=True)
    std  = np.std(features, axis=1, keepdims=True)
    return (features - mean) / (std + 1e-8)

def padder(features, target_len=44):
    if features.shape[1] < target_len:
        pad = target_len - features.shape[1]
        features = np.pad(features, ((0,0),(0,pad)))
    else:
        features = features[:, :target_len]
    return features
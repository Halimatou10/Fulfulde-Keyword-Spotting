import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("✅ Bibliothèques importées !")

# ── Charger votre fichier fulfulde ────────────
# On utilise directement Nagge.mpeg — pas besoin d'internet !
signal, sr = librosa.load("Nagge.mpeg", sr=16000)
print(f"✅ Fréquence    : {sr} Hz")
print(f"✅ Durée        : {len(signal)/sr:.2f} secondes")
print(f"✅ Échantillons : {len(signal)}")

# ── Waveform ──────────────────────────────────
plt.figure(figsize=(12, 3))
librosa.display.waveshow(signal, sr=sr, color='purple')
plt.title("Ma voix — Nagge (vache en fulfulde)")
plt.xlabel("Temps (secondes)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("notebooks/waveform_nagge.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ waveform_nagge.png sauvegardé !")

# ── FFT ───────────────────────────────────────
fft       = np.fft.fft(signal)
magnitude = np.abs(fft)
freqs     = np.fft.fftfreq(len(fft), 1/sr)
idx       = freqs > 0

plt.figure(figsize=(12, 3))
plt.plot(freqs[idx], magnitude[idx], color='teal')
plt.title("Spectre de Fréquences (FFT) — Nagge")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 4000)
plt.tight_layout()
plt.savefig("notebooks/fft_nagge.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ fft_nagge.png sauvegardé !")

# ── Spectrogramme ─────────────────────────────
stft           = librosa.stft(signal)
spectrogram_db = librosa.amplitude_to_db(np.abs(stft))

plt.figure(figsize=(12, 4))
librosa.display.specshow(
    spectrogram_db, sr=sr,
    x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title("Spectrogramme — Nagge")
plt.xlabel("Temps (secondes)")
plt.ylabel("Fréquence (Hz)")
plt.tight_layout()
plt.savefig("notebooks/spectrogramme_nagge.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ spectrogramme_nagge.png sauvegardé !")

# ── MFCC + Delta + Delta-Delta ────────────────
mfcc   = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
delta  = librosa.feature.delta(mfcc)
delta2 = librosa.feature.delta(mfcc, order=2)

fig, axes = plt.subplots(3, 1, figsize=(12, 9))

img1 = librosa.display.specshow(
    mfcc, sr=sr, x_axis='time',
    ax=axes[0], cmap='viridis')
axes[0].set_title("MFCC (40 coeff.) — Signature acoustique de NAGGE")
axes[0].set_ylabel("Coefficients")
fig.colorbar(img1, ax=axes[0])

img2 = librosa.display.specshow(
    delta, sr=sr, x_axis='time',
    ax=axes[1], cmap='plasma')
axes[1].set_title("Delta — Vitesse de changement")
axes[1].set_ylabel("Delta")
fig.colorbar(img2, ax=axes[1])

img3 = librosa.display.specshow(
    delta2, sr=sr, x_axis='time',
    ax=axes[2], cmap='magma')
axes[2].set_title("Delta-Delta — Accélération du changement")
axes[2].set_ylabel("Delta²")
fig.colorbar(img3, ax=axes[2])

plt.tight_layout()
plt.savefig("notebooks/mfcc_nagge.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ mfcc_nagge.png sauvegardé !")

# ── Résumé final ──────────────────────────────
features = np.concatenate([mfcc, delta, delta2], axis=0)
print("\n" + "="*45)
print("   RÉSUMÉ — JOUR 1 TERMINÉ")
print("="*45)
print(f"  Fichier     : Nagge.mpeg (fulfulde)")
print(f"  Durée       : {len(signal)/sr:.2f} secondes")
print(f"  Fréquence   : {sr} Hz")
print(f"  MFCC        : {mfcc.shape}")
print(f"  Features    : {features.shape}")
print(f"  Total       : {features.shape[0]*features.shape[1]} chiffres")
print("="*45)
print("\n✅ 4 images sauvegardées dans notebooks/")
print("✅ JOUR 1 TERMINÉ — Prêt pour samedi !")
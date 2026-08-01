import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# ── Charger le fichier audio ──────────────────
print("Chargement du fichier audio...")
signal, sr = librosa.load("Nagge.mpeg", sr=16000)

print(f"✅ Durée        : {len(signal)/sr:.2f} secondes")
print(f"✅ Fréquence    : {sr} Hz")
print(f"✅ Échantillons : {len(signal)}")

# ── Waveform ──────────────────────────────────
print("\nGénération de la waveform...")
plt.figure(figsize=(12, 3))
librosa.display.waveshow(signal, sr=sr, color='purple')
plt.title("Ma voix — Nagge (vache en fulfulde)")
plt.xlabel("Temps (secondes)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("waveform_nagge.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ waveform_nagge.png sauvegardé !")

# ── Extraction MFCC ───────────────────────────
print("\nExtraction des MFCC...")
mfcc   = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
delta  = librosa.feature.delta(mfcc)
delta2 = librosa.feature.delta(mfcc, order=2)

features = np.concatenate([mfcc, delta, delta2], axis=0)

print(f"✅ MFCC        : {mfcc.shape}")
print(f"✅ Delta       : {delta.shape}")
print(f"✅ Delta-Delta : {delta2.shape}")
print(f"✅ Features    : {features.shape}")

# ── Visualisation MFCC ────────────────────────
print("\nGénération des graphiques MFCC...")
fig, axes = plt.subplots(3, 1, figsize=(12, 9))

img1 = librosa.display.specshow(
    mfcc, sr=sr, x_axis='time', ax=axes[0], cmap='viridis')
axes[0].set_title("MFCC (40 coeff.) — Signature acoustique de NAGGE")
axes[0].set_ylabel("Coefficients")
fig.colorbar(img1, ax=axes[0])

img2 = librosa.display.specshow(
    delta, sr=sr, x_axis='time', ax=axes[1], cmap='plasma')
axes[1].set_title("Delta — Vitesse de changement")
axes[1].set_ylabel("Delta")
fig.colorbar(img2, ax=axes[1])

img3 = librosa.display.specshow(
    delta2, sr=sr, x_axis='time', ax=axes[2], cmap='magma')
axes[2].set_title("Delta-Delta — Accélération du changement")
axes[2].set_ylabel("Delta²")
fig.colorbar(img3, ax=axes[2])

plt.tight_layout()
plt.savefig("mfcc_nagge.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ mfcc_nagge.png sauvegardé !")

# ── Résumé final ──────────────────────────────
print("\n" + "=" * 45)
print("   RÉSUMÉ — FEATURES EXTRAITES DE NAGGE")
print("=" * 45)
print(f"  Mot-clé     : NAGGE (vache en fulfulde)")
print(f"  Durée       : {len(signal)/sr:.2f} secondes")
print(f"  Fréquence   : {sr} Hz")
print(f"  MFCC        : {mfcc.shape[0]} coefficients")
print(f"  Features    : {features.shape[0]} x {features.shape[1]}")
print(f"  Total       : {features.shape[0]*features.shape[1]} chiffres")
print("=" * 45)
print("\n✅ SUCCÈS — Prêt pour la démo de samedi !")
print("   Ouvrez waveform_nagge.png et mfcc_nagge.png")
print("   dans votre dossier KWS_Fulfulde")
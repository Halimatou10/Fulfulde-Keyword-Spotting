import sys
sys.path.append("src")
from features import extraire_mfcc, normaliser, padder

print("Test du pipeline complet...")

features, signal, sr = extraire_mfcc("data/raw/Nagge.mpeg")
print(f"✅ Features extraites : {features.shape}")

features_norm = normaliser(features)
print(f"✅ Features normalisées : {features_norm.shape}")

features_pad = padder(features_norm, target_len=44)
print(f"✅ Features paddées : {features_pad.shape}")

print("\n" + "="*40)
print("  PIPELINE COMPLET FONCTIONNEL !")
print("  Prêt pour l entrainement du modele")
print("="*40)
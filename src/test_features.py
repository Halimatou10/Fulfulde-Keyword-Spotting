import numpy as np

# Chemin vers le fichier .npy
fichier = "data/features/nagge/Nagge_features.npy"

# Chargement des features
features = np.load(fichier)

# Affichage des informations
print("✅ Fichier chargé avec succès !")
print("Shape :", features.shape)
print("Type :", features.dtype)

print("\nPremières valeurs :")
print(features[:, :5])
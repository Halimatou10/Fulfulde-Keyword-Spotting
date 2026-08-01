import os
import librosa
import numpy as np

print("✅ Bibliothèques importées !")
  

# ============================================================
# 1. CONFIGURATION DES DOSSIERS
# ============================================================

INPUT_FOLDER = "data/raw"
OUTPUT_FOLDER = "data/features"

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. FONCTION DE TRAITEMENT D'UN FICHIER AUDIO
# ============================================================

def traiter_audio(chemin_audio, chemin_sortie):
    """
    Cette fonction réalise tout le pipeline sur un fichier :

    1. Chargement à 16 kHz
    2. Normalisation de l'amplitude
    3. Suppression des silences
    4. Extraction MFCC
    5. Extraction Delta
    6. Extraction Delta²
    7. Concaténation
    8. Sauvegarde en .npy
    """

    print("\n" + "=" * 60)
    print("TRAITEMENT DU FICHIER")
    print("=" * 60)

    print(f"Fichier : {chemin_audio}")


    # ========================================================
    # ÉTAPE 1 — CHARGEMENT
    # ========================================================

    signal, sr = librosa.load(
        chemin_audio,
        sr=16000
    )

    print(
        f"Signal brut : "
        f"{len(signal)} échantillons — "
        f"{len(signal) / sr:.2f} secondes"
    )


    # ========================================================
    # ÉTAPE 2 — NORMALISATION
    # ========================================================

    max_amplitude = np.max(
        np.abs(signal)
    )

    if max_amplitude > 0:

        signal = signal / (
            max_amplitude + 1e-8
        )

    print(
        "Amplitude maximale après "
        f"normalisation : "
        f"{np.max(np.abs(signal)):.4f}"
    )


    # ========================================================
    # ÉTAPE 3 — SUPPRESSION DES SILENCES
    # ========================================================

    signal_clean, _ = librosa.effects.trim(
        signal,
        top_db=20
    )

    print(
        f"Signal nettoyé : "
        f"{len(signal_clean)} échantillons — "
        f"{len(signal_clean) / sr:.2f} secondes"
    )


    # ========================================================
    # ÉTAPE 4 — EXTRACTION MFCC
    # ========================================================

    mfcc = librosa.feature.mfcc(
        y=signal_clean,
        sr=sr,
        n_mfcc=13
    )

    print(
        f"MFCC : {mfcc.shape}"
    )


    # ========================================================
    # ÉTAPE 5 — DELTA
    # ========================================================

    delta = librosa.feature.delta(
        mfcc
    )

    print(
        f"Delta : {delta.shape}"
    )


    # ========================================================
    # ÉTAPE 6 — DELTA²
    # ========================================================

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    print(
        f"Delta² : {delta2.shape}"
    )


    # ========================================================
    # ÉTAPE 7 — CONCATÉNATION
    # ========================================================

    features = np.vstack([
        mfcc,
        delta,
        delta2
    ])

    print(
        f"Features finales : "
        f"{features.shape}"
    )


    # ========================================================
    # ÉTAPE 8 — SAUVEGARDE
    # ========================================================

    np.save(
        chemin_sortie,
        features
    )

    print(
        f"✅ Features sauvegardées : "
        f"{chemin_sortie}"
    )


# ============================================================
# 3. PARCOURS DU DATASET — BATCH
# ============================================================

print("\n")
print("=" * 60)
print("DÉBUT DU TRAITEMENT BATCH")
print("=" * 60)


nombre_fichiers = 0


# Parcourir les dossiers de mots
for mot in os.listdir(INPUT_FOLDER):

    chemin_mot = os.path.join(
        INPUT_FOLDER,
        mot
    )

    # Vérifier qu'il s'agit bien d'un dossier
    if not os.path.isdir(chemin_mot):
        continue


    # Créer le dossier features correspondant
    dossier_sortie = os.path.join(
        OUTPUT_FOLDER,
        mot
    )

    os.makedirs(
        dossier_sortie,
        exist_ok=True
    )


    # Parcourir les fichiers audio
    for fichier in os.listdir(chemin_mot):

        # Vérifier l'extension
        if not fichier.lower().endswith(
            (".wav", ".mp3", ".mpeg", ".mp4")
        ):
            continue


        # Chemin complet du fichier audio
        chemin_audio = os.path.join(
            chemin_mot,
            fichier
        )


        # Nom du fichier de sortie
        nom_features = (
            os.path.splitext(fichier)[0]
            + "_features.npy"
        )


        chemin_sortie = os.path.join(
            dossier_sortie,
            nom_features
        )


        # Traiter le fichier
        try:

            traiter_audio(
                chemin_audio,
                chemin_sortie
            )

            nombre_fichiers += 1

        except Exception as e:

            print(
                f"❌ Erreur avec "
                f"{chemin_audio}"
            )

            print(
                f"   Détail : {e}"
            )


# ============================================================
# 4. FIN DU PIPELINE
# ============================================================

print("\n")
print("=" * 60)
print("TRAITEMENT BATCH TERMINÉ")
print("=" * 60)

print(
    f"Nombre de fichiers traités : "
    f"{nombre_fichiers}"
)
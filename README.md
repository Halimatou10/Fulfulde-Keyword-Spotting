# KWS Fulfulde

## Description

Ce projet a pour objectif de développer un système de **Keyword Spotting (KWS)** pour la langue **fulfulde**.

Le système extrait des caractéristiques acoustiques (MFCC, Delta et Delta-Delta) à partir d'enregistrements audio afin de préparer l'entraînement d'un modèle de reconnaissance de mots-clés.

---

## Structure du projet

```
KWS_FULFULDE/
│
├── data/
│   ├── raw/           # Fichiers audio originaux
│   ├── features/      # Features extraites
│   └── augmented/     # Données augmentées
│
├── models/            # Modèles entraînés
│
├── notebooks/         # Images et visualisations
│
├── src/               # Code source
│
├── jour1_mfcc.py
├── jour2_mfcc.py
├── features.py
├── test_pipeline.py
│
├── requirements.txt
└── README.md
```

---

## Bibliothèques utilisées

- Python 3.14
- Librosa
- NumPy
- SciPy
- Matplotlib

---

## Fonctionnalités

- Chargement d'un fichier audio
- Visualisation de la forme d'onde
- Calcul de la FFT
- Génération du spectrogramme
- Extraction des MFCC
- Calcul des Delta
- Calcul des Delta-Delta

---

## Exemple d'exécution

```bash
python jour2_mfcc.py
```

---

## Résultats

Le programme génère automatiquement :

- waveform_nagge.png
- fft_nagge.png
- spectrogramme_nagge.png
- mfcc_nagge.png

---

## Objectif du projet

Construire un système de reconnaissance de mots-clés en fulfulde pour des applications de traitement automatique de la parole.

---

## Auteur

Ton Nom

Master Intelligence Artificielle

2026
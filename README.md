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
├── pipeline_batch.py
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
- Nettoyage des silences (VAD)
- Normalisation de l'amplitude
- Visualisation de la forme d'onde
- Calcul de la FFT
- Génération du spectrogramme
- Extraction des MFCC
- Calcul des Delta
- Calcul des Delta-Delta
- Sauvegarde des features au format `.npy`

---

## Exemple d'exécution

```bash
python pipeline_batch.py
```

---

## Résultats

Le programme génère automatiquement :

- waveform_nagge.png
- fft_nagge.png
- spectrogramme_nagge.png
- mfcc_nagge.png
- Nagge_features.npy

---

## Objectif du projet

Construire un système de reconnaissance de mots-clés en fulfulde pour des applications de traitement automatique de la parole, avec une première approche basée sur les MFCC et un classifieur de référence (baseline).

---

## Auteur

**Halimatou Sadia Ahmadou**

Master 2 — Systèmes Logiciels en Environnements Distribués

Université de Ngaoundéré

2026
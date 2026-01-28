# Segmentation Sémantique pour Conduite Autonome avec CARLA

Projet de segmentation sémantique automatique pour véhicules autonomes utilisant CARLA, PyTorch et les architectures ENet/U-Net.

## 📋 Table des matières

- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Workflow complet](#workflow-complet)
- [Utilisation](#utilisation)
- [Résultats attendus](#résultats-attendus)

## 🚀 Installation

### 1. Prérequis

- Python 3.8+
- CUDA 11.8+ (pour GPU)
- CARLA Simulator 0.9.13+

### 2. Installation des dépendances
```bash
# Cloner le projet
git clone <votre-repo>
cd projet_segmentation

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Installer CARLA
# Télécharger depuis https://github.com/carla-simulator/carla/releases
# Puis installer le package Python:
pip install carla==0.9.13
```

### 3. Configuration CARLA
```bash
# Lancer le serveur CARLA
cd /chemin/vers/CARLA
./CarlaUE4.sh  # Linux
# ou
CarlaUE4.exe  # Windows
```

## 📁 Structure du projet
```
projet_segmentation/
├── annotation_tool/          # Outil d'annotation manuelle
│   ├── annotator.py
│   └── label_config.json
├── models/                   # Architectures des modèles
│   ├── enet.py
│   └── unet.py
├── data/                     # Gestion des données
│   ├── dataset.py
│   └── augmentation.py
├── training/                 # Scripts d'entraînement
│   ├── train.py
│   ├── evaluate.py
│   └── config.py
├── utils/                    # Utilitaires
│   ├── metrics.py
│   └── visualization.py
├── carla_scripts/           # Scripts CARLA
│   ├── collect_images.py
│   └── test_realtime.py
├── checkpoints/             # Modèles sauvegardés
├── data/                    # Données
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── val/
│       ├── images/
│       └── masks/
└── requirements.txt
```

## 🔄 Workflow complet

### Étape 1: Collecte d'images depuis CARLA
```bash
# Lancer CARLA d'abord
python carla_scripts/collect_images.py \
    --output data/collected_images \
    --num_images 500 \
    --interval 10
```

### Étape 2: Annotation manuelle des images
```bash
python annotation_tool/annotator.py \
    --images data/collected_images \
    --output data/annotations \
    --config annotation_tool/label_config.json
```

**Contrôles de l'outil d'annotation:**
- Souris gauche: Dessiner
- Souris droite: Effacer
- 1-9: Sélectionner la classe
- n/p: Image suivante/précédente
- s: Sauvegarder
- z: Undo
- q: Quitter

### Étape 3: Organiser les données
```bash
# Créer la structure train/val
mkdir -p data/train/images data/train/masks
mkdir -p data/val/images data/val/masks

# Déplacer les images et masques annotés
# 80% pour train, 20% pour validation
```

### Étape 4: Entraînement du modèle
```bash
# Entraînement rapide (test)
python training/train.py \
    --config quick \
    --model unet \
    --experiment_name test_run

# Entraînement complet
python training/train.py \
    --config full \
    --model unet \
    --train_images data/train/images \
    --train_masks data/train/masks \
    --val_images data/val/images \
    --val_masks data/val/masks \
    --batch_size 8 \
    --epochs 100 \
    --experiment_name unet_full

# Avec ENet
python training/train.py \
    --config full \
    --model enet \
    --train_images data/train/images \
    --train_masks data/train/masks \
    --val_images data/val/images \
    --val_masks data/val/masks \
    --experiment_name enet_full
```

### Étape 5: Évaluation
```bash
# Évaluer sur le set de test
python training/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --images data/test/images \
    --masks data/test/masks \
    --output evaluation_results

# Prédiction sur une seule image
python training/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --images data/test/images \
    --masks data/test/masks \
    --single_image data/test/images/test_001.png
```

### Étape 6: Test en temps réel dans CARLA
```bash
# Lancer CARLA d'abord
python carla_scripts/test_realtime.py \
    --checkpoint checkpoints/best_model.pth \
    --width 800 \
    --height 600 \
    --save_video \
    --output results_video.avi
```

## 📊 Visualisation avec TensorBoard
```bash
# Pendant l'entraînement
tensorboard --logdir runs/

# Ouvrir http://localhost:6006 dans votre navigateur
```

## 🎯 Classes de segmentation



## 📈 Métriques d'évaluation

- **mIoU** (mean Intersection over Union): Métrique principale
- **Pixel Accuracy**: Précision globale
- **IoU par classe**: Performance par classe
- **Dice Coefficient**: Alternative à l'IoU

## 💡 Conseils

### Pour de meilleurs résultats:

1. **Annotation**: Annotez au moins 200-500 images pour un bon début
2. **Qualité > Quantité**: Mieux vaut moins d'images bien annotées
3. **Diversité**: Collectez des images dans différentes conditions (jour/nuit, météo, environnements)
4. **Augmentation**: Activez la data augmentation pendant l'entraînement
5. **Checkpoints**: Sauvegardez régulièrement vos modèles

### Débogage:
```bash
# Vérifier la distribution des classes
python -c "from utils.visualization import visualize_class_distribution; visualize_class_distribution('data/train/masks', 10)"

# Visualiser quelques prédictions
python training/evaluate.py --checkpoint checkpoints/best_model.pth ...
```

## 🐛 Problèmes courants

### CUDA out of memory
- Réduire `batch_size` dans la config
- Utiliser des images plus petites (resize_size)

### Mauvaise performance
- Vérifier la qualité des annotations
- Augmenter le nombre d'epochs
- Utiliser les class weights si déséquilibre

### CARLA ne se connecte pas
- Vérifier que le serveur CARLA est lancé
- Vérifier le port (défaut: 2000)

## 📝 Citation

Si vous utilisez ce code pour votre recherche, merci de citer:
```
@misc{segmentation_carla_2026,
  author = {Votre Nom},
  title = {Segmentation Sémantique pour Conduite Autonome avec CARLA},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/votre-repo}
}
```

## 📄 Licence

Ce projet est sous licence MIT.

## 🤝 Contribution

Les contributions sont les bienvenues! Pour contribuer:

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📧 Contact

Pour toute question: [votre.email@example.com]

---

**Bon courage pour votre projet! 🚗🤖**
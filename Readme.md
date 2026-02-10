# Segmentation Sémantique Binaire pour Conduite Autonome

Projet de segmentation sémantique **binaire** (Traversable vs. Obstrué) pour véhicules autonomes utilisant CARLA, PyTorch et l'architecture ENet.

## 🎯 Objectif

Le modèle apprend à **reconnaître et comprendre** tous les objets de la scène (voitures, piétons, routes, bâtiments, etc.) pour ensuite les classifier en **2 catégories** :
- 🟢 **Zone Traversable** (route, trottoir, terrain, etc.)
- 🔴 **Zone Obstruée** (voitures, piétons, bâtiments, etc.)

### Architecture du système

```
CARLA (23 classes sémantiques) 
    ↓
Collecte des données avec annotations multi-classes
    ↓
Entraînement du modèle (apprend les 23 classes)
    ↓
Conversion automatique vers 2 classes binaires
    ↓
Sortie finale : Traversable (vert) / Obstrué (rouge)
```

## 📋 Table des matières

- [Installation](#installation)
- [Gestion des dimensions](#-gestion-des-dimensions-important)
- [Collecte de données](#1-collecte-de-données-depuis-carla)
- [Préparation des données](#2-préparation-des-données)
- [Entraînement](#3-entraînement)
- [Évaluation](#4-évaluation)
- [Test temps réel](#5-test-en-temps-réel-dans-carla)
- [Structure du projet](#structure-du-projet)
- [Adaptation aux conditions extrêmes](#adaptation-aux-conditions-extrêmes)

## 📐 Gestion des Dimensions (Important!)

Le projet gère **intelligemment** les dimensions pour éviter les déformations d'image :

- **Collecte CARLA** : 800x600 pixels (par défaut)
- **Entraînement** : 512x512 avec **préservation de l'aspect ratio** (padding)
- **Avantage** : Pas de déformation des objets (voitures, piétons gardent leurs proportions)

**Configuration dans `config.py`** :
```python
PRESERVE_ASPECT_RATIO = True  # ← RECOMMANDÉ (utilise padding)
TRAINING_IMAGE_SIZE = (512, 512)  # Résolution du modèle
```

📖 **Guide complet** : Consultez [DIMENSIONS.md](DIMENSIONS.md) pour comprendre l'impact des résolutions et choisir la meilleure configuration.

## 📊 Workflow Complet

### Prérequis

- Python 3.8+
- CUDA 11.8+ (pour GPU, fortement recommandé)
- CARLA Simulator 0.9.13+
- 16GB RAM minimum
- GPU avec 6GB+ VRAM recommandé

### Installation des dépendances

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

# Installer CARLA Python API
pip install carla==0.9.13
```

### Configuration CARLA

```bash
# Télécharger CARLA depuis https://github.com/carla-simulator/carla/releases
# Extraire et lancer le serveur:

cd /chemin/vers/CARLA
./CarlaUE4.sh  # Linux
# ou
CarlaUE4.exe  # Windows
```

## 📊 Workflow Complet

### 1. Collecte de données depuis CARLA

Le script collecte automatiquement des images RGB avec leurs masques sémantiques (23 classes).

```bash
# Lancer CARLA d'abord !

# Collecte simple (500 images)
python carla_scripts/collect_images.py \
    --output data/collected \
    --num_images 500

# Collecte diversifiée (différentes météos)
python carla_scripts/collect_images.py \
    --output data/collected \
    --num_images 500 \
    --diverse

# Options avancées
python carla_scripts/collect_images.py \
    --output data/collected \
    --num_images 1000 \
    --diverse \
    --width 800 \
    --height 600 \
    --host localhost \
    --port 2000
```

**Sortie** :
- `data/collected/images/` : Images RGB
- `data/collected/masks/` : Masques sémantiques (.npy et .png)

### 2. Préparation des données

Organisez vos données en train/val/test (80/10/10 typiquement) :

```bash
# Créer la structure
mkdir -p data/train/images data/train/masks
mkdir -p data/val/images data/val/masks
mkdir -p data/test/images data/test/masks

# Déplacer les fichiers manuellement ou avec un script
# Exemple : 80% train, 10% val, 10% test
```

**Important** : La conversion multi-classe → binaire se fait **automatiquement** pendant l'entraînement !

### 3. Entraînement

#### 3.1 Entraînement rapide (test)

```bash
python training/train.py \
    --model enet \
    --train_images data/train/images \
    --train_masks data/train/masks \
    --val_images data/val/images \
    --val_masks data/val/masks \
    --epochs 10 \
    --batch_size 4 \
    --experiment_name test_run
```

#### 3.2 Entraînement complet

```bash
# Avec ENet (recommandé pour temps réel)
python training/train.py \
    --model enet \
    --train_images data/train/images \
    --train_masks data/train/masks \
    --val_images data/val/images \
    --val_masks data/val/masks \
    --epochs 100 \
    --batch_size 8 \
    --lr 5e-4 \
    --image_size 512 \
    --weighted_loss \
    --experiment_name enet_binary_v1

# Avec U-Net (meilleure précision)
python training/train.py \
    --model unet \
    --train_images data/train/images \
    --train_masks data/train/masks \
    --val_images data/val/images \
    --val_masks data/val/masks \
    --epochs 100 \
    --batch_size 4 \
    --experiment_name unet_binary_v1
```

#### 3.3 Reprendre un entraînement

```bash
python training/train.py \
    --model enet \
    --resume checkpoints/enet_binary_v1_last.pth \
    --epochs 150 \
    --experiment_name enet_binary_v1_continued
```

#### 3.4 Visualisation avec TensorBoard

```bash
tensorboard --logdir runs/
# Ouvrir http://localhost:6006
```

### 4. Évaluation

#### 4.1 Évaluation sur le dataset de test

```bash
python training/evaluate.py \
    --checkpoint checkpoints/enet_binary_v1_best.pth \
    --model enet \
    --images data/test/images \
    --masks data/test/masks \
    --batch_size 8 \
    --save_predictions \
    --output evaluation_results
```

**Sortie** :
- Métriques détaillées (mIoU, Pixel Accuracy, Dice)
- Visualisations dans `evaluation_results/`

#### 4.2 Évaluation sur une image unique

```bash
python training/evaluate.py \
    --checkpoint checkpoints/enet_binary_v1_best.pth \
    --model enet \
    --single_image data/test/images/test_001.png \
    --single_mask data/test/masks/test_001.npy \
    --output evaluation_results
```

#### 4.3 Comparaison de plusieurs modèles

```bash
python training/evaluate.py \
    --compare checkpoints/enet_binary_v1_best.pth \
              checkpoints/unet_binary_v1_best.pth \
    --compare_types enet unet \
    --images data/test/images \
    --masks data/test/masks
```

### 5. Test en temps réel dans CARLA

```bash
# Lancer CARLA d'abord !

# Test basique
python carla_scripts/test_realtime.py \
    --checkpoint checkpoints/enet_binary_v1_best.pth \
    --model enet

# Test avec enregistrement vidéo
python carla_scripts/test_realtime.py \
    --checkpoint checkpoints/enet_binary_v1_best.pth \
    --model enet \
    --save_video \
    --video_path results/demo.avi

# Options avancées
python carla_scripts/test_realtime.py \
    --checkpoint checkpoints/enet_binary_v1_best.pth \
    --model enet \
    --camera_width 1024 \
    --camera_height 768 \
    --display_width 1920 \
    --display_height 1080 \
    --save_video \
    --video_path results/demo_hd.avi
```

**Contrôles** :
- `q` : Quitter
- `s` : Sauvegarder la frame actuelle

## 📁 Structure du projet

```
projet_segmentation/
├── config.py                 # Configuration des classes et mapping binaire
├── requirements.txt          # Dépendances Python
│
├── models/                   # Architectures des modèles
│   ├── enet.py              # ENet (temps réel)
│   └── unet.py              # U-Net (précision)
│
├── data/                     # Gestion des données
│   └── dataset.py           # Dataset avec conversion automatique
│
├── training/                 # Scripts d'entraînement
│   ├── train.py             # Entraînement principal
│   └── evaluate.py          # Évaluation
│
├── utils/                    # Utilitaires
│   ├── metrics.py           # Calcul des métriques
│   └── visualization.py     # Visualisation
│
├── carla_scripts/           # Scripts CARLA
│   ├── collect_images.py   # Collecte de données
│   └── test_realtime.py    # Test temps réel
│
├── checkpoints/             # Modèles sauvegardés
├── runs/                    # Logs TensorBoard
└── data/                    # Données
    ├── train/
    │   ├── images/
    │   └── masks/
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/
```

## 🔄 Mapping des classes

Le fichier `config.py` définit le mapping des 23 classes CARLA vers 2 classes binaires :

### Classes CARLA → Binaire

**Traversable (0 - Vert)** :
- Road (route)
- RoadLine (ligne de route)
- Sidewalk (trottoir)
- Ground (sol)
- Bridge (pont)
- RailTrack (rail)
- Terrain (terrain)

**Obstrué (1 - Rouge)** :
- Building (bâtiment)
- Fence (barrière)
- Pedestrian (piéton)
- Pole (poteau)
- Vegetation (végétation)
- Vehicles (véhicules)
- Wall (mur)
- TrafficSign (panneau)
- Sky (ciel)
- TrafficLight (feu)
- Et autres obstacles

## 🌦️ Adaptation aux conditions extrêmes

### Phase 1 : Entraînement de base

Utilisez d'abord la collecte diversifiée pour obtenir des données dans différentes météos :

```bash
python carla_scripts/collect_images.py \
    --output data/diverse \
    --num_images 1000 \
    --diverse
```

### Phase 2 : Data augmentation intensive

Le dataset inclut déjà de l'augmentation pour simuler :
- ☁️ Brouillard
- 🌧️ Pluie
- ❄️ Neige
- 🌙 Conditions nocturnes
- 💨 Flou de mouvement
- 🔆 Variations de luminosité

Pour activer l'augmentation intensive pendant l'entraînement, modifiez `data/dataset.py` :

```python
from data.dataset import get_heavy_augmentation

# Dans train.py, remplacer get_training_augmentation par :
transform = get_heavy_augmentation(image_size)
```

### Phase 3 : Fine-tuning spécifique

Pour adapter à des conditions très spécifiques :

1. Collectez des données dans ces conditions dans CARLA
2. Fine-tunez le modèle pré-entraîné :

```bash
python training/train.py \
    --model enet \
    --resume checkpoints/enet_binary_v1_best.pth \
    --train_images data/extreme_conditions/images \
    --train_masks data/extreme_conditions/masks \
    --epochs 20 \
    --lr 1e-5 \
    --experiment_name enet_finetuned_extreme
```

## 📈 Métriques d'évaluation

Le système calcule :

- **mIoU** (mean Intersection over Union) : Métrique principale
- **Pixel Accuracy** : Précision globale
- **Dice Coefficient** : Alternative à l'IoU
- **IoU par classe** : Performance pour Traversable et Obstrué

### Objectifs de performance

- **mIoU > 0.85** : Excellent
- **mIoU > 0.80** : Très bon
- **mIoU > 0.75** : Bon
- **mIoU < 0.70** : Nécessite amélioration

## 💡 Conseils pour de meilleurs résultats

1. **Quantité de données** :
   - Minimum : 500 images
   - Recommandé : 2000-5000 images
   - Optimal : 10000+ images

2. **Diversité** :
   - Collectez dans différentes cartes CARLA
   - Variez les conditions météo
   - Incluez jour et nuit
   - Variez les scénarios (urbain, autoroute, rural)

3. **Équilibrage des classes** :
   - Utilisez `--weighted_loss` si déséquilibre
   - Vérifiez la distribution avec `visualize_class_distribution()`

4. **Optimisation** :
   - ENet : ~60 FPS sur GPU moderne (RTX 3060+)
   - U-Net : ~30 FPS sur GPU moderne
   - Augmentez `batch_size` si vous avez plus de VRAM

5. **Checkpoints** :
   - Sauvegardez régulièrement
   - Gardez le meilleur modèle (`_best.pth`)
   - Expérimentez avec différents hyperparamètres

## 🐛 Dépannage

### CUDA Out of Memory

```bash
# Réduire la taille du batch
--batch_size 2

# Réduire la taille des images
--image_size 256
```

### CARLA ne se connecte pas

```bash
# Vérifier que CARLA est lancé
ps aux | grep Carla

# Vérifier le port
--port 2000

# Augmenter le timeout dans le code si connexion lente
```

### Mauvaises performances

1. Vérifier la qualité des données
2. Augmenter le nombre d'epochs
3. Essayer différents learning rates
4. Utiliser `--weighted_loss`
5. Collecter plus de données

## 📧 Support

Pour toute question ou problème :
- Consultez les logs TensorBoard
- Vérifiez les métriques d'évaluation
- Visualisez les prédictions

## 📝 Citation

```bibtex
@misc{segmentation_carla_2026,
  title={Binary Semantic Segmentation for Autonomous Driving with CARLA},
  author={Votre Nom},
  year={2026},
  url={https://github.com/votre-repo}
}
```

## 📄 Licence

Ce projet est sous licence MIT.

---

**Bon courage pour votre projet! 🚗🤖**

Pour toute amélioration future, ce projet est conçu pour être facilement extensible. La séparation claire entre reconnaissance multi-classe et classification binaire permet d'adapter facilement le système à d'autres types de segmentation.
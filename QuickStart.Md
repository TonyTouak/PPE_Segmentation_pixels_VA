# 🚀 Guide de Démarrage Rapide

Guide pas-à-pas pour démarrer rapidement avec le projet de segmentation sémantique binaire.

## ⚡ Démarrage Express (30 minutes)

### ⚙️ Note sur les Dimensions

Le projet préserve automatiquement l'aspect ratio des images (pas de déformation) :
- CARLA collecte en 800x600
- Modèle entraîné en 512x512 (avec padding)
- **Résultat** : Objets aux proportions correctes ✓

📖 Voir [DIMENSIONS.md](DIMENSIONS.md) pour plus de détails.

### 1. Installation (5 min)

```bash
# Cloner et installer
git clone <votre-repo>
cd projet_segmentation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install carla==0.9.13

# Tester l'installation
python test_installation.py
```

### 2. Lancer CARLA (1 min)

```bash
cd /chemin/vers/CARLA
./CarlaUE4.sh  # ou CarlaUE4.exe sur Windows
```

### 3. Collecter des données (10 min)

```bash
# Terminal 2 (CARLA doit tourner)
python carla_scripts/collect_images.py \
    --output data/collected \
    --num_images 500 \
    --diverse
```

### 4. Préparer les données (2 min)

```bash
python utils/split_dataset.py \
    --images data/collected/images \
    --masks data/collected/masks \
    --output data
```

Vous aurez maintenant :
- `data/train/` (400 images)
- `data/val/` (50 images)  
- `data/test/` (50 images)

### 5. Entraîner le modèle (10 min pour test rapide)

```bash
# Test rapide (10 epochs)
python training/train.py \
    --model enet \
    --epochs 10 \
    --batch_size 4 \
    --experiment_name quick_test

# OU entraînement complet (2-3 heures)
python training/train.py \
    --model enet \
    --epochs 100 \
    --batch_size 8 \
    --weighted_loss \
    --experiment_name enet_full
```

### 6. Évaluer (2 min)

```bash
python training/evaluate.py \
    --checkpoint checkpoints/quick_test_best.pth \
    --model enet \
    --images data/test/images \
    --masks data/test/masks \
    --save_predictions
```

### 7. Tester en temps réel (temps réel!)

```bash
# CARLA doit tourner
python carla_scripts/test_realtime.py \
    --checkpoint checkpoints/quick_test_best.pth \
    --model enet
```

Appuyez sur `q` pour quitter, `s` pour sauvegarder une frame.

## 📊 Commandes Essentielles

### Collecte de données

```bash
# Basique (météo par défaut)
python carla_scripts/collect_images.py --output data/my_data --num_images 1000

# Diversifiée (5 météos différentes)
python carla_scripts/collect_images.py --output data/diverse --num_images 1000 --diverse

# Haute résolution
python carla_scripts/collect_images.py --output data/hd --num_images 500 --width 1920 --height 1080
```

### Entraînement

```bash
# ENet (rapide, temps réel)
python training/train.py --model enet --epochs 100 --batch_size 8 --experiment_name enet_v1

# U-Net (précis, plus lent)
python training/train.py --model unet --epochs 100 --batch_size 4 --experiment_name unet_v1

# U-Net Small (compromis)
python training/train.py --model unet_small --epochs 100 --batch_size 6 --experiment_name unet_small_v1
```

### Évaluation

```bash
# Dataset complet
python training/evaluate.py \
    --checkpoint checkpoints/enet_v1_best.pth \
    --model enet \
    --images data/test/images \
    --masks data/test/masks \
    --save_predictions

# Une seule image
python training/evaluate.py \
    --checkpoint checkpoints/enet_v1_best.pth \
    --model enet \
    --single_image data/test/images/test_001.png \
    --single_mask data/test/masks/test_001.npy
```

### Test temps réel

```bash
# Basique
python carla_scripts/test_realtime.py --checkpoint checkpoints/enet_v1_best.pth --model enet

# Avec vidéo
python carla_scripts/test_realtime.py \
    --checkpoint checkpoints/enet_v1_best.pth \
    --model enet \
    --save_video \
    --video_path results/demo.avi
```

## 🎯 Objectifs de Performance

Pour un bon modèle :

| Métrique | Objectif | Excellent |
|----------|----------|-----------|
| mIoU | > 0.75 | > 0.85 |
| Pixel Accuracy | > 0.90 | > 0.95 |
| FPS (GPU RTX 3060) | > 30 | > 60 |

## 🔧 Résolution de Problèmes Rapide

### Problème : CUDA Out of Memory

```bash
# Solution 1: Réduire batch size
--batch_size 2

# Solution 2: Réduire taille image
--image_size 256

# Solution 3: Utiliser U-Net Small
--model unet_small
```

### Problème : CARLA ne se connecte pas

```bash
# Vérifier que CARLA tourne
ps aux | grep Carla  # Linux
tasklist | findstr Carla  # Windows

# Essayer un autre port
--port 2001

# Vérifier l'hôte
--host 127.0.0.1
```

### Problème : Mauvaises métriques (mIoU < 0.5)

1. **Vérifier les données** :
   ```bash
   # Visualiser la distribution
   python -c "from utils.visualization import visualize_class_distribution; visualize_class_distribution('data/train/masks', 2)"
   ```

2. **Collecter plus de données** :
   - Minimum 1000 images recommandé
   - Varier les scénarios et météos

3. **Utiliser la loss pondérée** :
   ```bash
   --weighted_loss
   ```

4. **Augmenter les epochs** :
   ```bash
   --epochs 150
   ```

## 📈 Monitoring avec TensorBoard

```bash
# Pendant ou après l'entraînement
tensorboard --logdir runs/

# Ouvrir dans le navigateur
http://localhost:6006
```

Vous verrez :
- Courbes de loss (train/val)
- mIoU évolution
- Pixel Accuracy
- Learning rate

## 💾 Gestion des Checkpoints

Les checkpoints sont sauvegardés dans `checkpoints/` :

- `<experiment>_best.pth` : Meilleur modèle (mIoU max)
- `<experiment>_last.pth` : Dernier modèle

Pour reprendre un entraînement :

```bash
python training/train.py \
    --model enet \
    --resume checkpoints/enet_v1_last.pth \
    --epochs 150 \
    --experiment_name enet_v1_continued
```

## 🎨 Personnalisation

### Changer les couleurs de visualisation

Éditez `config.py` :

```python
BINARY_COLORS = {
    0: (0, 255, 0),    # Traversable (vert)
    1: (255, 0, 0)     # Obstrué (rouge)
}
```

### Changer le mapping des classes

Éditez `config.py` :

```python
CLASS_TO_BINARY = {
    # Exemple: Rendre le trottoir obstrué
    8: 1,  # Sidewalk -> Obstrué (au lieu de 0)
}
```

### Ajouter de l'augmentation personnalisée

Éditez `data/dataset.py` dans `get_training_augmentation()`.

## 📚 Ressources Supplémentaires

- **Documentation CARLA** : https://carla.readthedocs.io/
- **ENet Paper** : https://arxiv.org/abs/1606.02147
- **U-Net Paper** : https://arxiv.org/abs/1505.04597
- **PyTorch Docs** : https://pytorch.org/docs/

## 🆘 Aide

Si vous êtes bloqué :

1. Vérifiez les logs dans le terminal
2. Consultez TensorBoard pour les métriques
3. Visualisez quelques prédictions
4. Vérifiez que CARLA est bien lancé
5. Testez avec moins de données d'abord

## 🎓 Workflow Recommandé pour Débutants

1. **Jour 1** : Installation + test rapide (10 epochs)
2. **Jour 2** : Collecte de données diversifiées (2000+ images)
3. **Jour 3** : Entraînement complet (100 epochs)
4. **Jour 4** : Évaluation + optimisation
5. **Jour 5** : Tests temps réel + fine-tuning

---

**Bon développement! 🚗🤖**

Pour toute question, consultez le README principal ou les commentaires dans le code.
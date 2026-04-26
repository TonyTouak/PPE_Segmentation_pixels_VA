class TrainingConfig:
    pass

"""
Configuration des classes de segmentation
Mapping multi-classe CARLA -> Binaire (Traversable/Obstrué)
"""

# ============================================================================
# DEFINITION DES CLASSES CARLA (Source)
# ============================================================================
CARLA_CLASSES = {
    0: 'Unlabeled',
    1: 'Building',
    2: 'Fence',
    3: 'Other',
    4: 'Pedestrian',
    5: 'Pole',
    6: 'RoadLine',
    7: 'Road',
    8: 'Sidewalk',
    9: 'Vegetation',
    10: 'Vehicles',
    11: 'Wall',
    12: 'TrafficSign',
    13: 'Sky',
    14: 'Ground',
    15: 'Bridge',
    16: 'RailTrack',
    17: 'GuardRail',
    18: 'TrafficLight',
    19: 'Static',
    20: 'Dynamic',
    21: 'Water',
    22: 'Terrain'
}

# ============================================================================
# MAPPING BINAIRE (C'est ici que se joue la "Logique de conduite")
# ============================================================================
# 0 = Traversable (Vert : On peut rouler dessus)
# 1 = Obstrué (Rouge : Danger / Obstacle)

CLASS_TO_BINARY = {
    # ❌ OBSTACLES (Rouge = 1)
    0: 1,   # Unlabeled   → Rouge (inconnu = danger)
    1: 1,   # Building    → Rouge
    2: 1,   # Fence       → Rouge
    3: 1,   # Other       → Rouge (inconnu = danger)
    4: 1,   # Pedestrian  → Rouge ⚠️ CRITIQUE
    5: 1,   # Pole        → Rouge
    9: 1,   # Vegetation  → Rouge
    10: 1,  # Vehicles    → Rouge ⚠️ CRITIQUE
    11: 1,  # Wall        → Rouge
    12: 1,  # TrafficSign → Rouge
    13: 1,  # Sky         → Rouge
    17: 1,  # GuardRail   → Rouge
    18: 1,  # TrafficLight→ Rouge
    19: 1,  # Static      → Rouge
    20: 1,  # Dynamic     → Rouge
    21: 1,  # Water       → Rouge
    
    # ✅ SURFACES ROULABLES (Vert = 0)
    6: 0,   # RoadLine    → Vert (lignes de la route)
    7: 0,   # Road        → Vert ⚠️ ESSENTIEL
    8: 1,   # Sidewalk    → Vert (débattable : rouge pour voiture classique, vert pour véhicule d'urgence)
    14: 0,  # Ground      → Vert (terre battue, parking non pavé)
    15: 0,  # Bridge      → Vert (surface du pont)
    16: 0,  # RailTrack   → Vert (débattable : rouge si rails actifs)
    22: 0   # Terrain     → Vert (terrain plat, terre)
}

# ⚠️ NOTES SUR LE MAPPING :
# - Sidewalk (8) : VERT ici car permet au modèle d'éviter les obstacles sur le trottoir
#   Pour une vraie voiture, on pourrait le mettre en ROUGE (interdit)
# - RailTrack (16) : VERT car c'est une surface plate
#   Pour une vraie voiture, on devrait le mettre en ROUGE (rails = danger)
# - Ground/Terrain (14/22) : VERT car surfaces roulables en tout-terrain
#   Pour une voiture de ville, on pourrait les mettre en ROUGE

# Si vous voulez un mapping STRICT pour voiture de ville :
# Décommentez les lignes suivantes et commentez les lignes correspondantes ci-dessus
# CLASS_TO_BINARY_STRICT = {
#     8: 1,   # Sidewalk    → Rouge (interdit aux voitures)
#     14: 1,  # Ground      → Rouge (terre = instable)
#     16: 1,  # RailTrack   → Rouge (rails = danger)
#     22: 1   # Terrain     → Rouge (tout-terrain uniquement)
# }

# ============================================================================
# VISUALISATION ET DEBUG
# ============================================================================

# Classes binaires finales
BINARY_CLASSES = {
    0: 'Traversable',  # Vert
    1: 'Obstructed'    # Rouge
}

# Couleurs pour la visualisation (RGB)
BINARY_COLORS = {
    0: (0, 255, 0),    # Vert pur pour Traversable
    1: (255, 0, 0)     # Rouge pur pour Obstrué
}

# Couleurs CARLA pour visualisation intermédiaire (Debug seulement)
CARLA_COLORS = {
    0: (0, 0, 0),       # Unlabeled
    1: (70, 70, 70),    # Building
    2: (100, 40, 40),   # Fence
    3: (55, 90, 80),    # Other
    4: (220, 20, 60),   # Pedestrian
    5: (153, 153, 153), # Pole
    6: (157, 234, 50),  # RoadLine
    7: (128, 64, 128),  # Road
    8: (244, 35, 232),  # Sidewalk
    9: (107, 142, 35),  # Vegetation
    10: (0, 0, 142),    # Vehicles
    11: (102, 102, 156),# Wall
    12: (220, 220, 0),  # TrafficSign
    13: (70, 130, 180), # Sky
    14: (81, 0, 81),    # Ground
    15: (150, 100, 100),# Bridge
    16: (230, 150, 140),# RailTrack
    17: (180, 165, 180),# GuardRail
    18: (250, 170, 30), # TrafficLight
    19: (110, 190, 160),# Static
    20: (170, 120, 50), # Dynamic
    21: (45, 60, 150),  # Water
    22: (145, 170, 100) # Terrain
}

# ============================================================================
# PARAMETRES D'ENTRAINEMENT
# ============================================================================
NUM_CARLA_CLASSES = len(CARLA_CLASSES)  # 23 classes
NUM_BINARY_CLASSES = len(BINARY_CLASSES)  # 2 classes

# Poids des classes [Traversable, Obstrué]
# On met un poids plus fort sur l'obstacle pour éviter les collisions
CLASS_WEIGHTS = [1.0, 2.0]

# ============================================================================
# CONFIGURATION DES DIMENSIONS D'IMAGES
# ============================================================================

# Dimensions de collecte CARLA (par défaut)
CARLA_IMAGE_WIDTH = 800
CARLA_IMAGE_HEIGHT = 600

# Dimensions pour l'entraînement (résolution du modèle)
# Options communes: 
# - (512, 512): Bon compromis vitesse/qualité ✓ RECOMMANDÉ
# - (256, 256): Très rapide, moins précis
# - (768, 768): Plus précis, plus lent
# - (1024, 1024): Haute résolution, GPU puissant requis
TRAINING_IMAGE_SIZE = (512, 512)  # (H, W)

# Préserver l'aspect ratio (recommandé pour éviter les distorsions)
# Si True: utilise padding pour conserver les proportions
# Si False: redimensionne directement (peut déformer)
PRESERVE_ASPECT_RATIO = True

# Dimensions pour le test temps réel
REALTIME_CAMERA_WIDTH = 800
REALTIME_CAMERA_HEIGHT = 600
REALTIME_MODEL_SIZE = (512, 512)  # Résolution d'inférence du modèle
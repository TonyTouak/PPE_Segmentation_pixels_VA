"""
Configuration des classes de segmentation
Mapping multi-classe CARLA -> Binaire (Traversable/Obstrué)
"""

# Classes de segmentation sémantique CARLA (détection fine)
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

# Mapping vers classes binaires
# 0 = Traversable (vert), 1 = Obstrué (rouge)
CLASS_TO_BINARY = {
    0: 1,   # Unlabeled -> Obstrué
    1: 1,   # Building -> Obstrué
    2: 1,   # Fence -> Obstrué
    3: 1,   # Other -> Obstrué
    4: 1,   # Pedestrian -> Obstrué
    5: 1,   # Pole -> Obstrué
    6: 0,   # RoadLine -> Traversable
    7: 0,   # Road -> Traversable
    8: 0,   # Sidewalk -> Traversable
    9: 1,   # Vegetation -> Obstrué
    10: 1,  # Vehicles -> Obstrué
    11: 1,  # Wall -> Obstrué
    12: 1,  # TrafficSign -> Obstrué
    13: 1,  # Sky -> Obstrué
    14: 0,  # Ground -> Traversable
    15: 0,  # Bridge -> Traversable
    16: 0,  # RailTrack -> Traversable
    17: 1,  # GuardRail -> Obstrué
    18: 1,  # TrafficLight -> Obstrué
    19: 1,  # Static -> Obstrué
    20: 1,  # Dynamic -> Obstrué
    21: 1,  # Water -> Obstrué
    22: 0   # Terrain -> Traversable
}

# Classes binaires finales
BINARY_CLASSES = {
    0: 'Traversable',  # Vert
    1: 'Obstructed'    # Rouge
}

# Couleurs pour la visualisation (RGB)
BINARY_COLORS = {
    0: (0, 255, 0),    # Vert pour Traversable
    1: (255, 0, 0)     # Rouge pour Obstrué
}

# Couleurs CARLA pour visualisation intermédiaire (optionnel)
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

# Configuration pour l'entraînement
NUM_CARLA_CLASSES = len(CARLA_CLASSES)  # 23 classes
NUM_BINARY_CLASSES = len(BINARY_CLASSES)  # 2 classes

# Poids des classes pour gérer le déséquilibre (à ajuster selon vos données)
CLASS_WEIGHTS = [1.0, 2.0]  # [Traversable, Obstrué]
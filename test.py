import numpy as np
from PIL import Image
import os

# Remplace par le chemin d'un de tes masques CARLA (pas l'image RGB)
mask_path = "data/val/masks/F70-27.png" 
mask = np.array(Image.open(mask_path))

# CARLA stocke souvent la classe dans le canal R (Rouge)
if len(mask.shape) == 3:
    mask = mask[:, :, 0]

print(f"Classes trouvées dans le masque : {np.unique(mask)}")
# Si tu vois '7' dans la liste, la route est bien là !
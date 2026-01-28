"""
Dataset PyTorch pour la segmentation sémantique
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Optional, Callable, Tuple
import json


class SegmentationDataset(Dataset):
    """Dataset pour images et masques de segmentation"""
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[Callable] = None,
        mask_suffix: str = "_binary",
        num_classes: int = 2
    ):
        """
        Args:
            images_dir: Dossier contenant les images
            masks_dir: Dossier contenant les masques
            transform: Transformations à appliquer (doit gérer image ET masque)
            mask_suffix: Suffixe des fichiers masques
            num_classes: Nombre de classes de segmentation
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.mask_suffix = mask_suffix
        self.num_classes = num_classes
        
        # Trouver toutes les images
        self.images = sorted([
            f for f in self.images_dir.glob('*.png')
            if not f.stem.endswith(mask_suffix)
        ] + [
            f for f in self.images_dir.glob('*.jpg')
            if not f.stem.endswith(mask_suffix)
        ])
        
        # Vérifier que les masques existent
        self.valid_pairs = []
        for img_path in self.images:
            mask_path = self.masks_dir / f"{img_path.stem}{mask_suffix}.png"
            if mask_path.exists():
                self.valid_pairs.append((img_path, mask_path))
        
        print(f"Dataset initialisé: {len(self.valid_pairs)} paires image/masque trouvées")
        
        if len(self.valid_pairs) == 0:
            raise ValueError(f"Aucune paire image/masque trouvée dans {images_dir} et {masks_dir}")
    
    def __len__(self) -> int:
        return len(self.valid_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.valid_pairs[idx]
        
        # Charger l'image
        image = Image.open(img_path).convert('RGB')
        
        # Charger le masque
        mask = Image.open(mask_path)
        mask = np.array(mask, dtype=np.int64)
        mask = (mask > 127).astype(np.int64)
        # Vérifier les valeurs du masque
        if mask.max() >= self.num_classes:
            print(f"Warning: Masque {mask_path.name} contient des valeurs >= {self.num_classes}")
            mask = np.clip(mask, 0, self.num_classes - 1)
        
        # Appliquer les transformations
        if self.transform:
            # La transformation doit gérer à la fois l'image et le masque
            image, mask = self.transform(image, mask)
        else:
            # Conversion par défaut
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculer les poids des classes pour gérer le déséquilibre
        Utile pour la loss function
        """
        print("Calcul des poids des classes...")
        class_counts = np.zeros(2)
        
        for _, mask_path in self.valid_pairs:
            mask = np.array(Image.open(mask_path))
            for class_id in range(self.num_classes):
                class_counts[class_id] += np.sum(mask == class_id)
        
        # Éviter la division par zéro
        class_counts = np.maximum(class_counts, 1)
        
        # Poids inversement proportionnels à la fréquence
        total_pixels = class_counts.sum()
        class_weights = total_pixels / (self.num_classes * class_counts)
        
        print("Distribution des classes:")
        for i, (count, weight) in enumerate(zip(class_counts, class_weights)):
            print(f"  Classe {i}: {count:,} pixels ({count/total_pixels*100:.2f}%) - poids: {weight:.4f}")
        
        return torch.FloatTensor(class_weights)


def load_config(config_file: str) -> dict:
    """Charger la configuration des classes"""
    with open(config_file, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Test du dataset
    dataset = SegmentationDataset(
        images_dir="data/images",
        masks_dir="data/masks",
        num_classes=2
    )
    
    print(f"\nNombre d'échantillons: {len(dataset)}")
    
    if len(dataset) > 0:
        image, mask = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask unique values: {torch.unique(mask).tolist()}")
        
        # Calculer les poids
        weights = dataset.get_class_weights()
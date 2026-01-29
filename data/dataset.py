"""
Dataset personnalisé pour la segmentation sémantique
Gère le chargement et la conversion multi-classe -> binaire
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys

# Ajouter le chemin parent pour importer config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLASS_TO_BINARY, NUM_CARLA_CLASSES


class SegmentationDataset(Dataset):
    """
    Dataset pour segmentation sémantique avec conversion automatique
    Les masques peuvent être:
    - Multi-classe CARLA (0-22) -> automatiquement convertis en binaire
    - Déjà binaires (0-1)
    """
    
    def __init__(self, 
                 images_dir, 
                 masks_dir,
                 transform=None,
                 binary_output=True,
                 image_size=(512, 512)):
        """
        Args:
            images_dir: Dossier contenant les images RGB
            masks_dir: Dossier contenant les masques de segmentation
            transform: Transformations augmentation (albumentations)
            binary_output: Si True, convertit vers binaire (recommandé)
            image_size: Taille de redimensionnement (H, W)
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.binary_output = binary_output
        self.image_size = image_size
        
        # Lister les fichiers
        self.image_files = sorted([f for f in os.listdir(images_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Transformations de base
        if transform is None:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = transform
        
        print(f"Dataset initialisé avec {len(self.image_files)} images")
        print(f"Mode: {'Binaire' if binary_output else 'Multi-classe'}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Charger l'image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Charger le masque (peut être .png ou .npy)
        mask_name = os.path.splitext(img_name)[0]
        mask_path_png = os.path.join(self.masks_dir, f"{mask_name}.png")
        mask_path_npy = os.path.join(self.masks_dir, f"{mask_name}.npy")
        
        if os.path.exists(mask_path_png):
            mask = np.array(Image.open(mask_path_png))
        elif os.path.exists(mask_path_npy):
            mask = np.load(mask_path_npy)
        else:
            raise FileNotFoundError(f"Masque introuvable pour {img_name}")
        
        # Assurer que le masque est 2D
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # Conversion vers binaire si nécessaire
        if self.binary_output:
            mask = self._convert_to_binary(mask)
        
        # Appliquer les transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convertir le masque en tensor long
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()
        
        return image, mask
    
    def _convert_to_binary(self, mask):
        """
        Convertit un masque multi-classe CARLA en masque binaire
        0 = Traversable, 1 = Obstrué
        """
        binary_mask = np.zeros_like(mask, dtype=np.uint8)
        
        # Convertir chaque classe CARLA vers binaire
        for carla_class, binary_class in CLASS_TO_BINARY.items():
            binary_mask[mask == carla_class] = binary_class
        
        return binary_mask
    
    def get_sample_visualization(self, idx):
        """Retourne une image et son masque pour visualisation (sans normalisation)"""
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        mask_name = os.path.splitext(img_name)[0]
        mask_path_png = os.path.join(self.masks_dir, f"{mask_name}.png")
        mask_path_npy = os.path.join(self.masks_dir, f"{mask_name}.npy")
        
        if os.path.exists(mask_path_png):
            mask = np.array(Image.open(mask_path_png))
        else:
            mask = np.load(mask_path_npy)
        
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        if self.binary_output:
            mask = self._convert_to_binary(mask)
        
        return image, mask


def get_training_augmentation(image_size=(512, 512)):
    """
    Retourne les transformations d'augmentation pour l'entraînement
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        
        # Augmentations géométriques
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, 
                          rotate_limit=15, p=0.5),
        
        # Augmentations de couleur (simule différentes conditions)
        A.RandomBrightnessContrast(brightness_limit=0.2, 
                                  contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, 
                           sat_shift_limit=20, 
                           val_shift_limit=10, p=0.5),
        
        # Simulation de conditions météo
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        
        # Normalisation et conversion
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_validation_augmentation(image_size=(512, 512)):
    """
    Retourne les transformations pour la validation (sans augmentation)
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_heavy_augmentation(image_size=(512, 512)):
    """
    Augmentation intensive pour simuler conditions extrêmes
    (brouillard, pluie, neige, nuit)
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        
        # Géométrie
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, 
                          rotate_limit=20, p=0.6),
        
        # Conditions météo extrêmes
        A.OneOf([
            # Brouillard
            A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.8, p=1.0),
            # Pluie
            A.RandomRain(slant_lower=-10, slant_upper=10, 
                        drop_length=20, blur_value=3, p=1.0),
            # Neige
            A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, p=1.0),
        ], p=0.4),
        
        # Luminosité (jour/nuit)
        A.RandomBrightnessContrast(brightness_limit=0.3, 
                                  contrast_limit=0.3, p=0.7),
        
        # Ombres
        A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, p=0.3),
        
        # Bruit
        A.OneOf([
            A.GaussNoise(var_limit=(20, 80), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.4),
        
        # Flou
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=7, p=1.0),
        ], p=0.3),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


if __name__ == "__main__":
    # Test du dataset
    print("Test du SegmentationDataset")
    
    # Exemple de création (adapter les chemins)
    dataset = SegmentationDataset(
        images_dir="data/train/images",
        masks_dir="data/train/masks",
        transform=get_training_augmentation(),
        binary_output=True
    )
    
    print(f"Taille du dataset: {len(dataset)}")
    
    if len(dataset) > 0:
        image, mask = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask unique values: {torch.unique(mask)}")
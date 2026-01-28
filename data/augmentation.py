"""
Data augmentation pour la segmentation sémantique
Les transformations doivent être appliquées de manière cohérente à l'image ET au masque
"""

import torch
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image
from typing import Tuple


class SegmentationTransform:
    """Transformations synchronisées pour image et masque"""
    
    def __init__(
        self,
        resize_size: Tuple[int, int] = (512, 512),
        crop_size: Optional[Tuple[int, int]] = None,
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.0,
        rotation_degrees: float = 10,
        color_jitter: bool = True,
        normalize: bool = True
    ):
        """
        Args:
            resize_size: Taille de redimensionnement
            crop_size: Taille de crop aléatoire (None = pas de crop)
            horizontal_flip_prob: Probabilité de flip horizontal
            vertical_flip_prob: Probabilité de flip vertical
            rotation_degrees: Degrés max de rotation
            color_jitter: Appliquer color jitter
            normalize: Normaliser l'image
        """
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation_degrees = rotation_degrees
        self.color_jitter = color_jitter
        self.normalize = normalize
        
        # Normalisation ImageNet
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def __call__(self, image: Image.Image, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Appliquer les transformations
        
        Args:
            image: Image PIL
            mask: Masque numpy array
            
        Returns:
            image_tensor, mask_tensor
        """
        # Convertir le masque en PIL pour les transformations
        mask_pil = Image.fromarray(mask.astype(np.uint8))
        
        # Resize
        image = TF.resize(image, self.resize_size, interpolation=Image.BILINEAR)
        mask_pil = TF.resize(mask_pil, self.resize_size, interpolation=Image.NEAREST)
        
        # Random crop
        if self.crop_size is not None:
            i, j, h, w = self._get_random_crop_params(image, self.crop_size)
            image = TF.crop(image, i, j, h, w)
            mask_pil = TF.crop(mask_pil, i, j, h, w)
        
        # Random horizontal flip
        if random.random() < self.horizontal_flip_prob:
            image = TF.hflip(image)
            mask_pil = TF.hflip(mask_pil)
        
        # Random vertical flip
        if random.random() < self.vertical_flip_prob:
            image = TF.vflip(image)
            mask_pil = TF.vflip(mask_pil)
        
        # Random rotation
        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
            mask_pil = TF.rotate(mask_pil, angle, interpolation=Image.NEAREST)
        
        # Color jitter (seulement pour l'image)
        if self.color_jitter:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
        
        # Convertir en tensors
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask_pil, dtype=np.int64))
        mask = np.array(mask_pil, dtype=np.int64)
        mask = (mask > 127).astype(np.int64)  # 0 = obstrué, 1 = traversable
        mask = torch.from_numpy(mask)

                
        # Normalisation
        if self.normalize:
            image = TF.normalize(image, self.mean, self.std)
        
        return image, mask
    
    def _get_random_crop_params(self, image, output_size):
        """Obtenir les paramètres pour un crop aléatoire"""
        w, h = image.size
        th, tw = output_size
        
        if w == tw and h == th:
            return 0, 0, h, w
        
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw


class ValTransform:
    """Transformations pour la validation (pas d'augmentation)"""
    
    def __init__(
        self,
        resize_size: Tuple[int, int] = (512, 512),
        normalize: bool = True
    ):
        self.resize_size = resize_size
        self.normalize = normalize
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def __call__(self, image: Image.Image, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convertir le masque en PIL
        mask_pil = Image.fromarray(mask.astype(np.uint8))
        
        # Resize
        image = TF.resize(image, self.resize_size, interpolation=Image.BILINEAR)
        mask_pil = TF.resize(mask_pil, self.resize_size, interpolation=Image.NEAREST)
        
        # Convertir en tensors
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask_pil, dtype=np.int64))
        
        # Normalisation
        if self.normalize:
            image = TF.normalize(image, self.mean, self.std)
        
        return image, mask


if __name__ == "__main__":
    # Test des transformations
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Créer une image et un masque de test
    image = Image.new('RGB', (800, 600), color='red')
    mask = np.random.randint(0, 10, (600, 800))
    
    # Appliquer les transformations
    transform = SegmentationTransform(
        resize_size=(512, 512),
        horizontal_flip_prob=0.5,
        rotation_degrees=15
    )
    
    img_tensor, mask_tensor = transform(image, mask)
    
    print(f"Image tensor shape: {img_tensor.shape}")
    print(f"Mask tensor shape: {mask_tensor.shape}")
    print(f"Image range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    print(f"Mask unique values: {torch.unique(mask_tensor).tolist()}")
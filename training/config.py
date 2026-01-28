"""
Configuration centralisée pour l'entraînement
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class TrainingConfig:
    """Configuration pour l'entraînement"""
    
    # Données
    train_images_dir: str = "data/train/images"
    train_masks_dir: str = "data/train/masks"
    val_images_dir: str = "data/val/images"
    val_masks_dir: str = "data/val/masks"
    num_classes: int = 2
    #label_config: str = "annotation_tool/label_config.json"
    
    # Modèle
    model_name: str = "unet"  # "unet" ou "enet"
    model_params: dict = None
    pretrained: bool = False
    
    # Entraînement
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Optimiseur
    optimizer: str = "adam"  # "adam" ou "sgd"
    momentum: float = 0.9  # Pour SGD
    
    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # "cosine", "step", "plateau"
    lr_step_size: int = 30  # Pour StepLR
    lr_gamma: float = 0.1  # Pour StepLR
    
    # Loss
    use_class_weights: bool = True
    ignore_index: Optional[int] = None
    
    # Augmentation
    resize_size: Tuple[int, int] = (512, 512)
    crop_size: Optional[Tuple[int, int]] = None
    horizontal_flip_prob: float = 0.5
    rotation_degrees: float = 10
    color_jitter: bool = True
    
    # Training loop
    eval_every: int = 5  # Évaluer tous les N epochs
    save_every: int = 10  # Sauvegarder tous les N epochs
    print_every: int = 10  # Afficher les stats tous les N batches
    
    # Checkpoints
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None
    save_best_only: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging
    use_tensorboard: bool = True
    log_dir: str = "runs"
    experiment_name: str = "segmentation_exp"
    
    # Early stopping
    use_early_stopping: bool = True
    patience: int = 20
    min_delta: float = 0.001
    
    def __post_init__(self):
        if self.model_params is None:
            if self.model_name == "unet":
                self.model_params = {"base_channels": 64}
            elif self.model_name == "enet":
                self.model_params = {}


# Configurations prédéfinies
def get_quick_test_config() -> TrainingConfig:
    """Configuration pour un test rapide"""
    return TrainingConfig(
        batch_size=2,
        num_epochs=5,
        resize_size=(256, 256),
        eval_every=1,
        save_every=2,
        num_workers=0
    )


def get_full_training_config() -> TrainingConfig:
    """Configuration pour un entraînement complet"""
    return TrainingConfig(
        batch_size=16,
        num_epochs=150,
        learning_rate=5e-4,
        resize_size=(512, 512),
        use_scheduler=True,
        use_early_stopping=True,
        patience=25
    )


def get_high_quality_config() -> TrainingConfig:
    """Configuration pour un entraînement haute qualité"""
    return TrainingConfig(
        batch_size=8,
        num_epochs=200,
        learning_rate=3e-4,
        resize_size=(768, 768),
        crop_size=(512, 512),
        use_scheduler=True,
        scheduler_type="cosine",
        use_early_stopping=True,
        patience=30,
        horizontal_flip_prob=0.5,
        rotation_degrees=15,
        color_jitter=True
    )
"""
Métriques pour évaluer la segmentation sémantique
"""

import torch
import numpy as np
from typing import Dict, List


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Calculer l'Intersection over Union (IoU) pour chaque classe
    
    Args:
        pred: Prédictions (B, H, W) avec des indices de classes
        target: Ground truth (B, H, W) avec des indices de classes
        num_classes: Nombre de classes
        
    Returns:
        IoU tensor de shape (num_classes,)
    """
    ious = torch.zeros(num_classes)
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious[cls] = float('nan')  # Classe absente
        else:
            ious[cls] = intersection / union
    
    return ious


def compute_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """
    Calculer le mean IoU (mIoU)
    
    Args:
        pred: Prédictions (B, H, W)
        target: Ground truth (B, H, W)
        num_classes: Nombre de classes
        
    Returns:
        mIoU moyen sur toutes les classes présentes
    """
    ious = compute_iou(pred, target, num_classes)
    
    # Ignorer les classes absentes (NaN)
    valid_ious = ious[~torch.isnan(ious)]
    
    if len(valid_ious) == 0:
        return 0.0
    
    return valid_ious.mean().item()


def compute_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculer l'accuracy pixel par pixel
    
    Args:
        pred: Prédictions (B, H, W)
        target: Ground truth (B, H, W)
        
    Returns:
        Accuracy en pourcentage
    """
    correct = (pred == target).sum().item()
    total = target.numel()
    
    return 100.0 * correct / total


def compute_dice_coefficient(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Calculer le coefficient de Dice pour chaque classe
    
    Args:
        pred: Prédictions (B, H, W)
        target: Ground truth (B, H, W)
        num_classes: Nombre de classes
        
    Returns:
        Dice coefficient tensor de shape (num_classes,)
    """
    dice_scores = torch.zeros(num_classes)
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = (pred_inds & target_inds).sum().float()
        pred_sum = pred_inds.sum().float()
        target_sum = target_inds.sum().float()
        
        if pred_sum + target_sum == 0:
            dice_scores[cls] = float('nan')
        else:
            dice_scores[cls] = (2.0 * intersection) / (pred_sum + target_sum)
    
    return dice_scores


class MetricsTracker:
    """Classe pour suivre les métriques pendant l'entraînement"""
    
    def __init__(self, num_classes: int, class_names: List[str] = None):
        """
        Args:
            num_classes: Nombre de classes
            class_names: Noms des classes (optionnel)
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Réinitialiser les métriques"""
        self.iou_sum = torch.zeros(self.num_classes)
        self.dice_sum = torch.zeros(self.num_classes)
        self.class_counts = torch.zeros(self.num_classes)
        self.total_correct = 0
        self.total_pixels = 0
        self.num_batches = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Mettre à jour les métriques avec un nouveau batch
        
        Args:
            pred: Prédictions (B, C, H, W) logits ou (B, H, W) indices
            target: Ground truth (B, H, W)
        """
        # Si pred contient des logits, prendre l'argmax
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
        
        # Déplacer sur CPU pour les calculs
        pred = pred.cpu()
        target = target.cpu()
        
        # IoU
        ious = compute_iou(pred, target, self.num_classes)
        for cls in range(self.num_classes):
            if not torch.isnan(ious[cls]):
                self.iou_sum[cls] += ious[cls]
                self.class_counts[cls] += 1
        
        # Dice
        dice_scores = compute_dice_coefficient(pred, target, self.num_classes)
        for cls in range(self.num_classes):
            if not torch.isnan(dice_scores[cls]):
                self.dice_sum[cls] += dice_scores[cls]
        
        # Pixel accuracy
        correct = (pred == target).sum().item()
        total = target.numel()
        self.total_correct += correct
        self.total_pixels += total
        
        self.num_batches += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Obtenir les métriques moyennes
        
        Returns:
            Dictionnaire avec toutes les métriques
        """
        # IoU moyen par classe
        mean_ious = {}
        for cls in range(self.num_classes):
            if self.class_counts[cls] > 0:
                mean_ious[f"IoU/{self.class_names[cls]}"] = \
                    (self.iou_sum[cls] / self.class_counts[cls]).item()
        
        # mIoU global
        valid_ious = self.iou_sum[self.class_counts > 0] / self.class_counts[self.class_counts > 0]
        miou = valid_ious.mean().item() if len(valid_ious) > 0 else 0.0
        
        # Dice moyen par classe
        mean_dice = {}
        for cls in range(self.num_classes):
            if self.class_counts[cls] > 0:
                mean_dice[f"Dice/{self.class_names[cls]}"] = \
                    (self.dice_sum[cls] / self.class_counts[cls]).item()
        
        # Pixel accuracy
        pixel_acc = 100.0 * self.total_correct / self.total_pixels if self.total_pixels > 0 else 0.0
        
        metrics = {
            "mIoU": miou,
            "Pixel_Accuracy": pixel_acc,
            **mean_ious,
            **mean_dice
        }
        
        return metrics
    
    def print_metrics(self):
        """Afficher les métriques de manière formatée"""
        metrics = self.get_metrics()
        
        print("\n" + "="*50)
        print("MÉTRIQUES D'ÉVALUATION")
        print("="*50)
        print(f"mIoU: {metrics['mIoU']:.4f}")
        print(f"Pixel Accuracy: {metrics['Pixel_Accuracy']:.2f}%")
        print("\nIoU par classe:")
        for cls in range(self.num_classes):
            key = f"IoU/{self.class_names[cls]}"
            if key in metrics:
                print(f"  {self.class_names[cls]}: {metrics[key]:.4f}")
        print("="*50 + "\n")


if __name__ == "__main__":
    # Test des métriques
    num_classes = 5
    batch_size = 4
    height, width = 128, 128
    
    # Créer des prédictions et targets aléatoires
    pred = torch.randint(0, num_classes, (batch_size, height, width))
    target = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Calculer les métriques
    iou = compute_iou(pred, target, num_classes)
    miou = compute_miou(pred, target, num_classes)
    pixel_acc = compute_pixel_accuracy(pred, target)
    
    print(f"IoU par classe: {iou}")
    print(f"mIoU: {miou:.4f}")
    print(f"Pixel Accuracy: {pixel_acc:.2f}%")
    
    # Test du tracker
    tracker = MetricsTracker(num_classes, [f"Class_{i}" for i in range(num_classes)])
    tracker.update(pred, target)
    tracker.print_metrics()
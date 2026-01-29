"""
Métriques pour l'évaluation de la segmentation sémantique
- IoU (Intersection over Union)
- mIoU (mean IoU)
- Pixel Accuracy
- Dice Coefficient
"""

import numpy as np
import torch


class SegmentationMetrics:
    """Calcul des métriques de segmentation sémantique"""
    
    def __init__(self, num_classes=2, ignore_index=None):
        """
        Args:
            num_classes: Nombre de classes
            ignore_index: Index de classe à ignorer (optionnel)
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Réinitialise les compteurs"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, predictions, targets):
        """
        Met à jour la matrice de confusion
        
        Args:
            predictions: Tensor de prédictions (B, H, W) ou (B, C, H, W)
            targets: Tensor de targets (B, H, W)
        """
        # Convertir en numpy
        if torch.is_tensor(predictions):
            if predictions.ndim == 4:  # (B, C, H, W)
                predictions = torch.argmax(predictions, dim=1)
            predictions = predictions.cpu().numpy()
        
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        # Aplatir
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Ignorer les pixels spécifiés
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            predictions = predictions[mask]
            targets = targets[mask]
        
        # Assurer que les valeurs sont dans les limites
        mask = (targets >= 0) & (targets < self.num_classes)
        predictions = predictions[mask]
        targets = targets[mask]
        
        # Mise à jour de la matrice de confusion
        for pred, target in zip(predictions, targets):
            self.confusion_matrix[target, pred] += 1
    
    def get_confusion_matrix(self):
        """Retourne la matrice de confusion"""
        return self.confusion_matrix
    
    def get_iou_per_class(self):
        """
        Calcule l'IoU pour chaque classe
        
        Returns:
            iou_per_class: Array des IoU par classe
        """
        iou_per_class = np.zeros(self.num_classes)
        
        for i in range(self.num_classes):
            # Intersection
            intersection = self.confusion_matrix[i, i]
            
            # Union
            union = (self.confusion_matrix[i, :].sum() + 
                    self.confusion_matrix[:, i].sum() - 
                    self.confusion_matrix[i, i])
            
            if union == 0:
                iou_per_class[i] = float('nan')
            else:
                iou_per_class[i] = intersection / union
        
        return iou_per_class
    
    def get_miou(self):
        """
        Calcule le mIoU (mean IoU)
        
        Returns:
            miou: Mean IoU sur toutes les classes
        """
        iou_per_class = self.get_iou_per_class()
        
        # Ignorer les NaN (classes non présentes)
        valid_ious = iou_per_class[~np.isnan(iou_per_class)]
        
        if len(valid_ious) == 0:
            return 0.0
        
        return np.mean(valid_ious)
    
    def get_pixel_accuracy(self):
        """
        Calcule la précision pixel globale
        
        Returns:
            pixel_accuracy: Pourcentage de pixels correctement classifiés
        """
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        
        if total == 0:
            return 0.0
        
        return correct / total
    
    def get_dice_per_class(self):
        """
        Calcule le coefficient de Dice pour chaque classe
        
        Returns:
            dice_per_class: Array des coefficients de Dice par classe
        """
        dice_per_class = np.zeros(self.num_classes)
        
        for i in range(self.num_classes):
            # Intersection
            intersection = self.confusion_matrix[i, i]
            
            # Somme des prédictions et targets pour cette classe
            pred_sum = self.confusion_matrix[:, i].sum()
            target_sum = self.confusion_matrix[i, :].sum()
            
            denominator = pred_sum + target_sum
            
            if denominator == 0:
                dice_per_class[i] = float('nan')
            else:
                dice_per_class[i] = (2 * intersection) / denominator
        
        return dice_per_class
    
    def get_mean_dice(self):
        """
        Calcule le coefficient de Dice moyen
        
        Returns:
            mean_dice: Mean Dice sur toutes les classes
        """
        dice_per_class = self.get_dice_per_class()
        
        # Ignorer les NaN
        valid_dice = dice_per_class[~np.isnan(dice_per_class)]
        
        if len(valid_dice) == 0:
            return 0.0
        
        return np.mean(valid_dice)
    
    def get_class_accuracy(self):
        """
        Calcule la précision pour chaque classe
        
        Returns:
            accuracy_per_class: Array des précisions par classe
        """
        accuracy_per_class = np.zeros(self.num_classes)
        
        for i in range(self.num_classes):
            total = self.confusion_matrix[i, :].sum()
            
            if total == 0:
                accuracy_per_class[i] = float('nan')
            else:
                accuracy_per_class[i] = self.confusion_matrix[i, i] / total
        
        return accuracy_per_class
    
    def get_summary(self):
        """
        Retourne un résumé complet des métriques
        
        Returns:
            summary: Dictionnaire avec toutes les métriques
        """
        iou_per_class = self.get_iou_per_class()
        dice_per_class = self.get_dice_per_class()
        accuracy_per_class = self.get_class_accuracy()
        
        summary = {
            'mIoU': self.get_miou(),
            'pixel_accuracy': self.get_pixel_accuracy(),
            'mean_dice': self.get_mean_dice(),
            'iou_per_class': iou_per_class,
            'dice_per_class': dice_per_class,
            'accuracy_per_class': accuracy_per_class,
            'confusion_matrix': self.confusion_matrix
        }
        
        return summary
    
    def print_summary(self, class_names=None):
        """
        Affiche un résumé formaté des métriques
        
        Args:
            class_names: Liste des noms de classes (optionnel)
        """
        if class_names is None:
            class_names = [f"Class {i}" for i in range(self.num_classes)]
        
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("RÉSUMÉ DES MÉTRIQUES")
        print("="*60)
        
        print(f"\nMétriques globales:")
        print(f"  mIoU:            {summary['mIoU']:.4f}")
        print(f"  Pixel Accuracy:  {summary['pixel_accuracy']:.4f}")
        print(f"  Mean Dice:       {summary['mean_dice']:.4f}")
        
        print(f"\nMétriques par classe:")
        print(f"{'Classe':<20} {'IoU':<10} {'Dice':<10} {'Accuracy':<10}")
        print("-"*60)
        
        for i, name in enumerate(class_names):
            iou = summary['iou_per_class'][i]
            dice = summary['dice_per_class'][i]
            acc = summary['accuracy_per_class'][i]
            
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            dice_str = f"{dice:.4f}" if not np.isnan(dice) else "N/A"
            acc_str = f"{acc:.4f}" if not np.isnan(acc) else "N/A"
            
            print(f"{name:<20} {iou_str:<10} {dice_str:<10} {acc_str:<10}")
        
        print("\nMatrice de confusion:")
        print(summary['confusion_matrix'])
        print("="*60 + "\n")


def compute_iou(pred, target, num_classes=2, ignore_index=None):
    """
    Fonction utilitaire pour calculer l'IoU rapidement
    
    Args:
        pred: Prédictions (Tensor ou numpy array)
        target: Ground truth (Tensor ou numpy array)
        num_classes: Nombre de classes
        ignore_index: Index à ignorer
    
    Returns:
        iou: IoU moyen
    """
    metrics = SegmentationMetrics(num_classes, ignore_index)
    metrics.update(pred, target)
    return metrics.get_miou()


if __name__ == "__main__":
    # Test des métriques
    print("Test des métriques")
    
    # Créer des données de test
    num_classes = 2
    pred = torch.randint(0, num_classes, (4, 256, 256))
    target = torch.randint(0, num_classes, (4, 256, 256))
    
    # Calculer les métriques
    metrics = SegmentationMetrics(num_classes=num_classes)
    metrics.update(pred, target)
    
    # Afficher le résumé
    metrics.print_summary(class_names=['Traversable', 'Obstructed'])
"""
Utilitaires pour visualiser les résultats
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from typing import List, Tuple


def visualize_prediction(image: np.ndarray, ground_truth: np.ndarray, 
                        prediction: np.ndarray, class_colors: dict,
                        class_names: List[str] = None,
                        save_path: str = None):
    """
    Visualiser une prédiction avec matplotlib
    
    Args:
        image: Image originale (H, W, 3)
        ground_truth: Ground truth (H, W)
        prediction: Prédiction (H, W)
        class_colors: Dictionnaire {class_id: (R, G, B)}
        class_names: Noms des classes
        save_path: Chemin de sauvegarde (optionnel)
    """
    # Convertir les masques en couleur
    gt_color = mask_to_color(ground_truth, class_colors)
    pred_color = mask_to_color(prediction, class_colors)
    
    # Créer la figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Image originale
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Image Originale', fontsize=14)
    axes[0, 0].axis('off')
    
    # Ground truth
    axes[0, 1].imshow(gt_color)
    axes[0, 1].set_title('Ground Truth', fontsize=14)
    axes[0, 1].axis('off')
    
    # Prédiction
    axes[1, 0].imshow(pred_color)
    axes[1, 0].set_title('Prédiction', fontsize=14)
    axes[1, 0].axis('off')
    
    # Overlay
    overlay = cv2.addWeighted(image, 0.6, pred_color, 0.4, 0)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay', fontsize=14)
    axes[1, 1].axis('off')
    
    # Ajouter la légende
    if class_names:
        legend_elements = []
        for class_id, color in class_colors.items():
            if class_id < len(class_names):
                from matplotlib.patches import Patch
                legend_elements.append(
                    Patch(facecolor=np.array(color)/255.0, 
                          label=class_names[class_id])
                )
        
        fig.legend(handles=legend_elements, loc='center right', 
                  fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualisation sauvegardée: {save_path}")
    
    plt.show()


def mask_to_color(mask: np.ndarray, class_colors: dict) -> np.ndarray:
    """
    Convertir un masque de classes en image RGB colorée
    
    Args:
        mask: Masque (H, W) avec indices de classes
        class_colors: Dictionnaire {class_id: (R, G, B)}
        
    Returns:
        Image RGB colorée (H, W, 3)
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in class_colors.items():
        color_mask[mask == class_id] = color
    
    return color_mask


def plot_training_history(train_losses: List[float], val_mious: List[float],
                         save_path: str = None):
    """
    Tracer l'historique d'entraînement
    
    Args:
        train_losses: Liste des pertes d'entraînement
        val_mious: Liste des mIoU de validation
        save_path: Chemin de sauvegarde (optionnel)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # mIoU
    axes[1].plot(val_mious, label='Validation mIoU', color='orange', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('mIoU', fontsize=12)
    axes[1].set_title('Validation mIoU', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Historique sauvegardé: {save_path}")
    
    plt.show()


def visualize_class_distribution(dataset_dir: str, num_classes: int,
                                 class_names: List[str] = None,
                                 save_path: str = None):
    """
    Visualiser la distribution des classes dans un dataset
    
    Args:
        dataset_dir: Dossier contenant les masques
        num_classes: Nombre de classes
        class_names: Noms des classes
        save_path: Chemin de sauvegarde (optionnel)
    """
    from PIL import Image
    
    masks_dir = Path(dataset_dir)
    mask_files = list(masks_dir.glob('*_mask.png'))
    
    print(f"Analyse de {len(mask_files)} masques...")
    
    class_counts = np.zeros(num_classes)
    
    for mask_file in mask_files:
        mask = np.array(Image.open(mask_file))
        for class_id in range(num_classes):
            class_counts[class_id] += np.sum(mask == class_id)
    
    # Calculer les pourcentages
    total_pixels = class_counts.sum()
    class_percentages = 100.0 * class_counts / total_pixels
    
    # Tracer
    plt.figure(figsize=(12, 6))
    
    x_labels = class_names if class_names else [f"Class {i}" for i in range(num_classes)]
    x_pos = np.arange(num_classes)
    
    bars = plt.bar(x_pos, class_percentages, color='steelblue', alpha=0.7)
    
    # Colorer les barres en fonction du pourcentage
    for i, bar in enumerate(bars):
        if class_percentages[i] < 1:
            bar.set_color('red')
        elif class_percentages[i] < 5:
            bar.set_color('orange')
    
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Pourcentage (%)', fontsize=12)
    plt.title('Distribution des Classes dans le Dataset', fontsize=14)
    plt.xticks(x_pos, x_labels, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs sur les barres
    for i, (count, pct) in enumerate(zip(class_counts, class_percentages)):
        plt.text(i, pct + 0.5, f'{pct:.1f}%\n({int(count):,})', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Distribution sauvegardée: {save_path}")
    
    plt.show()
    
    # Afficher les statistiques
    print("\n" + "="*60)
    print("DISTRIBUTION DES CLASSES")
    print("="*60)
    for i, name in enumerate(x_labels):
        print(f"{name:20s}: {class_counts[i]:12,.0f} pixels ({class_percentages[i]:5.2f}%)")
    print("="*60)
    print(f"{'Total':20s}: {total_pixels:12,.0f} pixels")
    print("="*60 + "\n")


def compare_models(checkpoint_paths: List[str], metric_name: str = 'mIoU',
                  model_names: List[str] = None, save_path: str = None):
    """
    Comparer plusieurs modèles entraînés
    
    Args:
        checkpoint_paths: Liste des chemins vers les checkpoints
        metric_name: Nom de la métrique à comparer
        model_names: Noms des modèles
        save_path: Chemin de sauvegarde (optionnel)
    """
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(checkpoint_paths))]
    
    metrics = []
    
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'metrics' in checkpoint:
            metric_value = checkpoint['metrics'].get(metric_name, 0)
        else:
            metric_value = checkpoint.get('best_miou', 0)
        metrics.append(metric_value)
    
    # Tracer
    plt.figure(figsize=(10, 6))
    
    x_pos = np.arange(len(model_names))
    bars = plt.bar(x_pos, metrics, color='steelblue', alpha=0.7)
    
    # Colorer la meilleure barre
    best_idx = np.argmax(metrics)
    bars[best_idx].set_color('green')
    
    plt.xlabel('Modèles', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'Comparaison des Modèles - {metric_name}', fontsize=14)
    plt.xticks(x_pos, model_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs
    for i, metric in enumerate(metrics):
        plt.text(i, metric + 0.01, f'{metric:.4f}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparaison sauvegardée: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test de visualisation
    print("Test des fonctions de visualisation...")
    
    # Créer des données de test
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    gt = np.random.randint(0, 5, (512, 512), dtype=np.uint8)
    pred = np.random.randint(0, 5, (512, 512), dtype=np.uint8)
    
    class_colors = {
        0: (0, 0, 0),
        1: (128, 64, 128),
        2: (244, 35, 232),
        3: (70, 70, 70),
        4: (107, 142, 35)
    }
    
    class_names = ['Background', 'Road', 'Sidewalk', 'Building', 'Vegetation']
    
    visualize_prediction(image, gt, pred, class_colors, class_names)
"""
Utilitaires de visualisation pour la segmentation sémantique
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import cv2
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BINARY_COLORS, BINARY_CLASSES


def visualize_predictions(images, masks, predictions, num_samples=4, save_path=None):
    """
    Visualise les images, masques ground truth et prédictions côte à côte
    
    Args:
        images: Tensor d'images (B, C, H, W) normalisées
        masks: Tensor de masques ground truth (B, H, W)
        predictions: Tensor de prédictions (B, H, W) ou (B, C, H, W)
        num_samples: Nombre d'échantillons à visualiser
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    # Dénormaliser les images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    if torch.is_tensor(images):
        images = images.cpu()
        images = images * std + mean
        images = torch.clamp(images, 0, 1)
        images = images.permute(0, 2, 3, 1).numpy()
    
    if torch.is_tensor(masks):
        masks = masks.cpu().numpy()
    
    if torch.is_tensor(predictions):
        if predictions.ndim == 4:  # (B, C, H, W)
            predictions = torch.argmax(predictions, dim=1)
        predictions = predictions.cpu().numpy()
    
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Image originale
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title('Image Originale')
        axes[i, 0].axis('off')
        
        # Ground truth
        gt_colored = colorize_mask(masks[i])
        axes[i, 1].imshow(gt_colored)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prédiction
        pred_colored = colorize_mask(predictions[i])
        axes[i, 2].imshow(pred_colored)
        axes[i, 2].set_title('Prédiction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée: {save_path}")
    else:
        plt.show()
    
    plt.close()


def colorize_mask(mask, class_colors=None):
    """
    Convertit un masque de classes en image RGB colorée
    
    Args:
        mask: Masque (H, W) avec indices de classes
        class_colors: Dictionnaire {classe: (R, G, B)} (optionnel)
    
    Returns:
        colored_mask: Image RGB (H, W, 3)
    """
    if class_colors is None:
        class_colors = BINARY_COLORS
    
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx, color in class_colors.items():
        colored_mask[mask == class_idx] = color
    
    return colored_mask


def overlay_prediction_on_image(image, prediction, alpha=0.5):
    """
    Superpose la prédiction sur l'image originale
    
    Args:
        image: Image RGB (H, W, 3)
        prediction: Masque de prédiction (H, W)
        alpha: Transparence de l'overlay (0-1)
    
    Returns:
        overlayed: Image avec overlay
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    
    # Assurer que l'image est en uint8
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    
    # Coloriser la prédiction
    pred_colored = colorize_mask(prediction)
    
    # Superposer
    overlayed = cv2.addWeighted(image, 1 - alpha, pred_colored, alpha, 0)
    
    return overlayed


def visualize_batch_predictions(model, dataloader, device, num_batches=1, save_dir=None):
    """
    Visualise les prédictions pour plusieurs batches
    
    Args:
        model: Modèle entraîné
        dataloader: DataLoader
        device: Device (cuda/cpu)
        num_batches: Nombre de batches à visualiser
        save_dir: Dossier pour sauvegarder les figures
    """
    model.eval()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, f'batch_{batch_idx}.png')
            
            visualize_predictions(
                images.cpu(), 
                masks.cpu(), 
                predictions.cpu(),
                num_samples=min(4, len(images)),
                save_path=save_path
            )


def create_comparison_video(images, masks, predictions, output_path, fps=10):
    """
    Crée une vidéo de comparaison entre ground truth et prédictions
    
    Args:
        images: List ou array d'images
        masks: List ou array de masques
        predictions: List ou array de prédictions
        output_path: Chemin de sortie pour la vidéo
        fps: Frames per second
    """
    if len(images) == 0:
        print("Pas d'images à traiter")
        return
    
    # Préparer le writer vidéo
    h, w = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w * 3, h))
    
    for img, mask, pred in zip(images, masks, predictions):
        # Coloriser
        mask_colored = colorize_mask(mask)
        pred_colored = colorize_mask(pred)
        
        # Convertir BGR pour OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mask_bgr = cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR)
        pred_bgr = cv2.cvtColor(pred_colored, cv2.COLOR_RGB2BGR)
        
        # Concaténer horizontalement
        frame = np.hstack([img_bgr, mask_bgr, pred_bgr])
        
        writer.write(frame)
    
    writer.release()
    print(f"Vidéo créée: {output_path}")


def plot_training_curves(log_file, save_path=None):
    """
    Trace les courbes d'entraînement depuis un fichier de log
    
    Args:
        log_file: Fichier contenant les métriques d'entraînement
        save_path: Chemin pour sauvegarder la figure
    """
    # Cette fonction nécessite un fichier de log structuré
    # Implémentation à adapter selon le format de vos logs
    pass


def visualize_class_distribution(masks_dir, num_classes=2, sample_size=100):
    """
    Visualise la distribution des classes dans le dataset
    
    Args:
        masks_dir: Dossier contenant les masques
        num_classes: Nombre de classes
        sample_size: Nombre de masques à échantillonner
    """
    mask_files = [f for f in os.listdir(masks_dir) 
                  if f.endswith(('.png', '.npy'))][:sample_size]
    
    class_counts = np.zeros(num_classes)
    
    for mask_file in mask_files:
        mask_path = os.path.join(masks_dir, mask_file)
        
        if mask_file.endswith('.npy'):
            mask = np.load(mask_path)
        else:
            mask = np.array(Image.open(mask_path))
        
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        for class_idx in range(num_classes):
            class_counts[class_idx] += np.sum(mask == class_idx)
    
    # Normaliser
    class_counts = class_counts / class_counts.sum() * 100
    
    # Tracer
    plt.figure(figsize=(10, 6))
    class_names = [BINARY_CLASSES[i] for i in range(num_classes)]
    colors = [np.array(BINARY_COLORS[i]) / 255.0 for i in range(num_classes)]
    
    plt.bar(class_names, class_counts, color=colors)
    plt.ylabel('Pourcentage (%)')
    plt.title('Distribution des classes dans le dataset')
    plt.ylim([0, 100])
    
    for i, v in enumerate(class_counts):
        plt.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def save_prediction_as_image(prediction, output_path, colorize=True):
    """
    Sauvegarde une prédiction comme image
    
    Args:
        prediction: Masque de prédiction (H, W)
        output_path: Chemin de sortie
        colorize: Si True, colorise le masque, sinon sauvegarde les indices
    """
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    
    if colorize:
        colored = colorize_mask(prediction)
        Image.fromarray(colored).save(output_path)
    else:
        Image.fromarray(prediction.astype(np.uint8)).save(output_path)
    
    print(f"Prédiction sauvegardée: {output_path}")


if __name__ == "__main__":
    # Test de visualisation
    print("Test des fonctions de visualisation")
    
    # Créer des données de test
    images = torch.rand(2, 3, 256, 256)
    masks = torch.randint(0, 2, (2, 256, 256))
    predictions = torch.randint(0, 2, (2, 256, 256))
    
    # Visualiser
    visualize_predictions(images, masks, predictions, num_samples=2)
"""
Script d'évaluation pour modèle de segmentation
Calcule les métriques détaillées sur le dataset de test
"""

import os
import sys
import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

# Ajouter le chemin parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.enet import get_enet_model
from models.unet import get_unet_model
from data.dataset import SegmentationDataset, get_validation_augmentation
from utils.metrics import SegmentationMetrics
from utils.visualization import visualize_predictions, visualize_batch_predictions
from config import BINARY_CLASSES


def evaluate_model(model, dataloader, device, save_predictions=False, output_dir=None):
    """
    Évalue le modèle sur un dataset
    
    Args:
        model: Modèle à évaluer
        dataloader: DataLoader du dataset de test
        device: Device (cuda/cpu)
        save_predictions: Sauvegarder les prédictions
        output_dir: Dossier de sortie
    
    Returns:
        metrics: Dictionnaire des métriques
    """
    model.eval()
    metrics = SegmentationMetrics(num_classes=2)
    
    all_images = []
    all_masks = []
    all_predictions = []
    
    print("\nÉvaluation en cours...")
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            masks = masks.to(device)
            
            # Prédiction
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # Mise à jour des métriques
            metrics.update(predictions, masks)
            
            # Sauvegarder pour visualisation
            if save_predictions and batch_idx < 10:  # Sauvegarder les 10 premiers batches
                all_images.extend(images.cpu())
                all_masks.extend(masks.cpu())
                all_predictions.extend(predictions.cpu())
    
    # Afficher le résumé
    print("\n" + "="*60)
    print("RÉSULTATS DE L'ÉVALUATION")
    print("="*60)
    
    metrics.print_summary(class_names=list(BINARY_CLASSES.values()))
    
    # Sauvegarder les visualisations
    if save_predictions and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualiser par groupes de 4
        for i in range(0, min(len(all_images), 40), 4):
            end_idx = min(i + 4, len(all_images))
            save_path = os.path.join(output_dir, f'predictions_{i//4:03d}.png')
            
            visualize_predictions(
                torch.stack(all_images[i:end_idx]),
                torch.stack(all_masks[i:end_idx]),
                torch.stack(all_predictions[i:end_idx]),
                num_samples=end_idx - i,
                save_path=save_path
            )
        
        print(f"\nVisualisations sauvegardées dans {output_dir}")
    
    return metrics.get_summary()


def evaluate_single_image(model, image_path, mask_path, device, save_path=None):
    """
    Évalue le modèle sur une seule image
    
    Args:
        model: Modèle à évaluer
        image_path: Chemin de l'image
        mask_path: Chemin du masque (optionnel)
        device: Device
        save_path: Chemin pour sauvegarder la visualisation
    """
    from PIL import Image
    import torchvision.transforms as T
    
    model.eval()
    
    # Charger l'image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Transformation
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Prédiction
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).squeeze(0)
    
    # Charger le masque si disponible
    mask = None
    if mask_path and os.path.exists(mask_path):
        if mask_path.endswith('.npy'):
            mask = np.load(mask_path)
        else:
            mask = np.array(Image.open(mask_path))
        
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        mask = torch.from_numpy(mask)
        
        # Calculer les métriques
        metrics = SegmentationMetrics(num_classes=2)
        metrics.update(prediction.unsqueeze(0), mask.unsqueeze(0))
        
        print("\n" + "="*60)
        print("MÉTRIQUES POUR L'IMAGE")
        print("="*60)
        metrics.print_summary(class_names=list(BINARY_CLASSES.values()))
    
    # Visualiser
    if mask is not None:
        visualize_predictions(
            input_tensor.cpu(),
            mask.unsqueeze(0),
            prediction.unsqueeze(0),
            num_samples=1,
            save_path=save_path
        )
    else:
        # Visualiser seulement l'image et la prédiction
        import matplotlib.pyplot as plt
        from utils.visualization import colorize_mask
        
        # Dénormaliser l'image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_denorm = input_tensor * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        img_denorm = img_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        pred_colored = colorize_mask(prediction.cpu().numpy())
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_denorm)
        axes[0].set_title('Image Originale')
        axes[0].axis('off')
        
        axes[1].imshow(pred_colored)
        axes[1].set_title('Prédiction')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualisation sauvegardée: {save_path}")
        else:
            plt.show()
        
        plt.close()


def compare_models(checkpoint_paths, model_types, dataloader, device):
    """
    Compare plusieurs modèles sur le même dataset
    
    Args:
        checkpoint_paths: Liste des chemins de checkpoints
        model_types: Liste des types de modèles
        dataloader: DataLoader du dataset de test
        device: Device
    """
    results = {}
    
    for checkpoint_path, model_type in zip(checkpoint_paths, model_types):
        print(f"\n{'='*60}")
        print(f"Évaluation: {os.path.basename(checkpoint_path)}")
        print(f"{'='*60}")
        
        # Charger le modèle
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if model_type == 'enet':
            model = get_enet_model(num_classes=2)
        elif model_type == 'unet':
            model = get_unet_model(num_classes=2, model_type='standard')
        elif model_type == 'unet_small':
            model = get_unet_model(num_classes=2, model_type='small')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Évaluer
        metrics = evaluate_model(model, dataloader, device, save_predictions=False)
        
        results[os.path.basename(checkpoint_path)] = metrics
    
    # Afficher la comparaison
    print("\n" + "="*60)
    print("COMPARAISON DES MODÈLES")
    print("="*60)
    print(f"{'Modèle':<30} {'mIoU':<10} {'Pixel Acc':<12} {'Dice':<10}")
    print("-"*60)
    
    for name, metrics in results.items():
        print(f"{name:<30} {metrics['mIoU']:.4f}    {metrics['pixel_accuracy']:.4f}      {metrics['mean_dice']:.4f}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Évaluation du modèle de segmentation')
    
    # Modèle
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Chemin vers le checkpoint du modèle')
    parser.add_argument('--model', type=str, default='enet',
                       choices=['enet', 'unet', 'unet_small'],
                       help='Type de modèle')
    
    # Données
    parser.add_argument('--images', type=str, default='data/test/images',
                       help='Dossier des images de test')
    parser.add_argument('--masks', type=str, default='data/test/masks',
                       help='Dossier des masques de test')
    
    # Évaluation
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Taille du batch')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Sauvegarder les visualisations des prédictions')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Dossier de sortie')
    
    # Image unique
    parser.add_argument('--single_image', type=str, default=None,
                       help='Évaluer sur une seule image')
    parser.add_argument('--single_mask', type=str, default=None,
                       help='Masque pour l\'image unique (optionnel)')
    
    # Comparaison
    parser.add_argument('--compare', nargs='+', default=None,
                       help='Liste de checkpoints à comparer')
    parser.add_argument('--compare_types', nargs='+', default=None,
                       help='Types de modèles correspondants')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Mode: image unique
    if args.single_image:
        print(f"\nÉvaluation sur une seule image: {args.single_image}")
        
        # Charger le modèle
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        if args.model == 'enet':
            model = get_enet_model(num_classes=2)
        elif args.model == 'unet':
            model = get_unet_model(num_classes=2, model_type='standard')
        elif args.model == 'unet_small':
            model = get_unet_model(num_classes=2, model_type='small')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        save_path = os.path.join(args.output, 'single_prediction.png')
        os.makedirs(args.output, exist_ok=True)
        
        evaluate_single_image(
            model, args.single_image, args.single_mask,
            device, save_path=save_path
        )
        
        return
    
    # Mode: comparaison de modèles
    if args.compare:
        if args.compare_types is None or len(args.compare) != len(args.compare_types):
            print("Erreur: --compare_types doit avoir la même longueur que --compare")
            return
        
        # Créer le dataloader
        dataset = SegmentationDataset(
            images_dir=args.images,
            masks_dir=args.masks,
            transform=get_validation_augmentation((512, 512)),
            binary_output=True
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        compare_models(args.compare, args.compare_types, dataloader, device)
        
        return
    
    # Mode: évaluation standard
    print(f"\nChargement du modèle depuis {args.checkpoint}...")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if args.model == 'enet':
        model = get_enet_model(num_classes=2)
    elif args.model == 'unet':
        model = get_unet_model(num_classes=2, model_type='standard')
    elif args.model == 'unet_small':
        model = get_unet_model(num_classes=2, model_type='small')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print("Modèle chargé avec succès!")
    
    # Créer le dataset et dataloader
    print(f"\nChargement du dataset de test...")
    
    dataset = SegmentationDataset(
        images_dir=args.images,
        masks_dir=args.masks,
        transform=get_validation_augmentation((512, 512)),
        binary_output=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset de test: {len(dataset)} images")
    
    # Évaluer
    evaluate_model(
        model, dataloader, device,
        save_predictions=args.save_predictions,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
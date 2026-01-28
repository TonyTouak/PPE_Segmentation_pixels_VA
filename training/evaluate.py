"""
Script d'évaluation pour tester un modèle entraîné
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import cv2
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.enet import ENet
from models.unet import UNet
from data.dataset import SegmentationDataset, load_config
from data.augmentation import ValTransform
from utils.metrics import MetricsTracker, compute_iou, compute_miou


class Evaluator:
    """Classe pour évaluer un modèle entraîné"""
    
    def __init__(self, checkpoint_path: str, images_dir: str, masks_dir: str, 
                 output_dir: str = None, device: str = None):
        """
        Args:
            checkpoint_path: Chemin vers le checkpoint du modèle
            images_dir: Dossier contenant les images de test
            masks_dir: Dossier contenant les masques de test
            output_dir: Dossier pour sauvegarder les visualisations (optionnel)
            device: Device à utiliser
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Device: {self.device}")
        print(f"Chargement du checkpoint: {checkpoint_path}")
        
        # Charger le checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint['config']
        self.class_names = checkpoint['class_names']
        
        # Charger la configuration des couleurs
        label_config = load_config(self.config.label_config)
        self.class_colors = {cls['id']: tuple(cls['color']) for cls in label_config['classes']}
        
        # Créer le modèle
        if self.config.model_name.lower() == "unet":
            self.model = UNet(
                num_classes=self.config.num_classes,
                **self.config.model_params
            )
        elif self.config.model_name.lower() == "enet":
            self.model = ENet(
                num_classes=self.config.num_classes,
                **self.config.model_params
            )
        
        # Charger les poids
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Modèle chargé: {self.config.model_name}")
        print(f"Epoch: {checkpoint['epoch']}")
        print(f"Best mIoU: {checkpoint.get('best_miou', 'N/A')}")
        
        # Créer le dataset de test
        val_transform = ValTransform(
            resize_size=self.config.resize_size,
            normalize=True
        )
        
        self.test_dataset = SegmentationDataset(
            images_dir=images_dir,
            masks_dir=masks_dir,
            transform=val_transform,
            num_classes=self.config.num_classes
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        print(f"Dataset de test: {len(self.test_dataset)} échantillons\n")
    
    @torch.no_grad()
    def evaluate(self, save_predictions: bool = True):
        """Évaluer le modèle sur le dataset de test"""
        print("Évaluation en cours...")
        
        metrics_tracker = MetricsTracker(
            num_classes=self.config.num_classes,
            class_names=self.class_names
        )
        
        for batch_idx, (images, masks) in enumerate(tqdm(self.test_loader)):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Prédiction
            outputs = self.model(images)
            predictions = outputs.argmax(dim=1)
            
            # Mettre à jour les métriques
            metrics_tracker.update(predictions, masks)
            
            # Sauvegarder les visualisations
            if save_predictions and self.output_dir:
                self._save_predictions(images, masks, predictions, batch_idx)
        
        # Afficher les résultats
        print("\n" + "="*60)
        print("RÉSULTATS D'ÉVALUATION")
        print("="*60)
        metrics_tracker.print_metrics()
        
        return metrics_tracker.get_metrics()
    
    def _save_predictions(self, images: torch.Tensor, masks: torch.Tensor, 
                         predictions: torch.Tensor, batch_idx: int):
        """Sauvegarder les visualisations des prédictions"""
        batch_size = images.size(0)
        
        for i in range(batch_size):
            # Dénormaliser l'image
            img = images[i].cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Convertir les masques en couleur
            mask_color = self._mask_to_color(masks[i].cpu().numpy())
            pred_color = self._mask_to_color(predictions[i].cpu().numpy())
            
            # Créer une visualisation combinée
            h, w = img.shape[:2]
            combined = np.zeros((h, w*3, 3), dtype=np.uint8)
            combined[:, :w] = img
            combined[:, w:2*w] = mask_color
            combined[:, 2*w:] = pred_color
            
            # Ajouter des labels
            cv2.putText(combined, "Image", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Ground Truth", (w+10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Prediction", (2*w+10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Sauvegarder
            idx = batch_idx * images.size(0) + i
            output_path = self.output_dir / f"prediction_{idx:04d}.png"
            cv2.imwrite(str(output_path), combined)
    
    def _mask_to_color(self, mask: np.ndarray) -> np.ndarray:
        """Convertir un masque de classes en image RGB colorée"""
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in self.class_colors.items():
            color_mask[mask == class_id] = color
        
        return cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    
    @torch.no_grad()
    def predict_single_image(self, image_path: str, save_path: str = None):
        """
        Faire une prédiction sur une seule image
        
        Args:
            image_path: Chemin vers l'image
            save_path: Chemin de sauvegarde (optionnel)
        """
        from PIL import Image
        
        # Charger l'image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Appliquer les transformations
        transform = ValTransform(
            resize_size=self.config.resize_size,
            normalize=True
        )
        
        # Transform attend une image et un masque, on crée un masque vide
        dummy_mask = np.zeros((image.size[1], image.size[0]), dtype=np.int64)
        image_tensor, _ = transform(image, dummy_mask)
        
        # Ajouter la dimension batch
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Prédiction
        output = self.model(image_tensor)
        prediction = output.argmax(dim=1).squeeze(0).cpu().numpy()
        
        # Redimensionner à la taille originale
        prediction = cv2.resize(
            prediction.astype(np.uint8),
            original_size,
            interpolation=cv2.INTER_NEAREST
        )
        
        # Convertir en couleur
        prediction_color = self._mask_to_color(prediction)
        
        if save_path:
            cv2.imwrite(save_path, prediction_color)
            print(f"Prédiction sauvegardée: {save_path}")
        
        return prediction, prediction_color


def main():
    parser = argparse.ArgumentParser(description="Évaluation de modèle de segmentation")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Chemin vers le checkpoint du modèle')
    parser.add_argument('--images', type=str, required=True,
                       help='Dossier contenant les images de test')
    parser.add_argument('--masks', type=str, required=True,
                       help='Dossier contenant les masques de test')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Dossier de sortie pour les visualisations')
    parser.add_argument('--device', type=str, default=None,
                       help='Device à utiliser (cuda/cpu)')
    parser.add_argument('--single_image', type=str, default=None,
                       help='Prédire sur une seule image')
    
    args = parser.parse_args()
    
    # Créer l'évaluateur
    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        images_dir=args.images,
        masks_dir=args.masks,
        output_dir=args.output,
        device=args.device
    )
    
    # Évaluation ou prédiction simple
    if args.single_image:
        output_path = Path(args.output) / f"{Path(args.single_image).stem}_prediction.png"
        evaluator.predict_single_image(args.single_image, str(output_path))
    else:
        metrics = evaluator.evaluate(save_predictions=True)
        print(f"\nVisualisations sauvegardées dans: {args.output}")


if __name__ == "__main__":
    main()
"""
Script d'entraînement pour segmentation sémantique binaire
Supporte ENet et U-Net avec conversion automatique multi-classe -> binaire
"""

import os
import sys
import argparse
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Ajouter le chemin parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.enet import get_enet_model
from models.unet import get_unet_model
from data.dataset import SegmentationDataset, get_training_augmentation, get_validation_augmentation
from config import NUM_BINARY_CLASSES, CLASS_WEIGHTS
from utils.metrics import SegmentationMetrics
from utils.visualization import visualize_predictions


class Trainer:
    """Entraîneur pour segmentation sémantique"""
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 device, experiment_name, checkpoint_dir='checkpoints'):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.experiment_name = experiment_name
        self.checkpoint_dir = checkpoint_dir
        
        # TensorBoard
        self.writer = SummaryWriter(f'runs/{experiment_name}')
        
        # Métriques
        self.metrics = SegmentationMetrics(num_classes=NUM_BINARY_CLASSES)
        
        # Meilleur score
        self.best_miou = 0.0
        
        # Créer le dossier de checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Entraînement: {experiment_name}")
        print(f"Device: {device}")
        print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
        print(f"{'='*60}\n")
    
    def train_epoch(self, epoch):
        """Entraîne le modèle pour une epoch"""
        self.model.train()
        epoch_loss = 0.0
        self.metrics.reset()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Métriques
            epoch_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            self.metrics.update(preds, masks)
            
            # Mise à jour de la barre de progression
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculer les métriques moyennes
        avg_loss = epoch_loss / len(self.train_loader)
        miou = self.metrics.get_miou()
        pixel_acc = self.metrics.get_pixel_accuracy()
        
        return avg_loss, miou, pixel_acc
    
    def validate(self, epoch):
        """Validation du modèle"""
        self.model.eval()
        epoch_loss = 0.0
        self.metrics.reset()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
            
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Métriques
                epoch_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                self.metrics.update(preds, masks)
                
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculer les métriques moyennes
        avg_loss = epoch_loss / len(self.val_loader)
        miou = self.metrics.get_miou()
        pixel_acc = self.metrics.get_pixel_accuracy()
        
        return avg_loss, miou, pixel_acc
    
    def save_checkpoint(self, epoch, miou, is_best=False):
        """Sauvegarde un checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'miou': miou,
            'experiment_name': self.experiment_name
        }
        
        # Sauvegarder le dernier modèle
        path = os.path.join(self.checkpoint_dir, f'{self.experiment_name}_last.pth')
        torch.save(checkpoint, path)
        
        # Sauvegarder le meilleur modèle
        if is_best:
            path = os.path.join(self.checkpoint_dir, f'{self.experiment_name}_best.pth')
            torch.save(checkpoint, path)
            print(f"✓ Meilleur modèle sauvegardé (mIoU: {miou:.4f})")
    
    def train(self, num_epochs, save_every=5):
        """Entraînement complet"""
        print(f"Début de l'entraînement pour {num_epochs} epochs\n")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Entraînement
            train_loss, train_miou, train_acc = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_miou, val_acc = self.validate(epoch)
            
            epoch_time = time.time() - start_time
            
            # Logging
            print(f"\nEpoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s")
            print(f"Train - Loss: {train_loss:.4f} | mIoU: {train_miou:.4f} | Acc: {train_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f} | mIoU: {val_miou:.4f} | Acc: {val_acc:.4f}")
            
            # TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('mIoU/train', train_miou, epoch)
            self.writer.add_scalar('mIoU/val', val_miou, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            # Sauvegarder checkpoint
            is_best = val_miou > self.best_miou
            if is_best:
                self.best_miou = val_miou
            
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_miou, is_best)
            
            print()
        
        print(f"\n{'='*60}")
        print(f"Entraînement terminé!")
        print(f"Meilleur mIoU: {self.best_miou:.4f}")
        print(f"{'='*60}\n")
        
        self.writer.close()


def get_dataloader(images_dir, masks_dir, batch_size, image_size, 
                   is_train=True, num_workers=4):
    """Crée un DataLoader"""
    
    if is_train:
        transform = get_training_augmentation(image_size)
        shuffle = True
    else:
        transform = get_validation_augmentation(image_size)
        shuffle = False
    
    dataset = SegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=transform,
        binary_output=True,
        image_size=image_size
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


def main():
    parser = argparse.ArgumentParser(description='Entraînement segmentation sémantique')
    
    # Données
    parser.add_argument('--train_images', type=str, default='data/train/images',
                       help='Dossier des images d\'entraînement')
    parser.add_argument('--train_masks', type=str, default='data/train/masks',
                       help='Dossier des masques d\'entraînement')
    parser.add_argument('--val_images', type=str, default='data/val/images',
                       help='Dossier des images de validation')
    parser.add_argument('--val_masks', type=str, default='data/val/masks',
                       help='Dossier des masques de validation')
    
    # Modèle
    parser.add_argument('--model', type=str, default='enet', choices=['enet', 'unet', 'unet_small'],
                       help='Architecture du modèle')
    
    # Entraînement
    parser.add_argument('--epochs', type=int, default=100,
                       help='Nombre d\'epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Taille du batch')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Taille des images (carré)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Nombre de workers pour le DataLoader')
    
    # Checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Dossier de sauvegarde des checkpoints')
    parser.add_argument('--experiment_name', type=str, default='experiment',
                       help='Nom de l\'expérience')
    parser.add_argument('--resume', type=str, default=None,
                       help='Chemin vers un checkpoint à reprendre')
    
    # Options
    parser.add_argument('--weighted_loss', action='store_true',
                       help='Utiliser une loss pondérée pour les classes')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Créer les DataLoaders
    print("\nChargement des données...")
    image_size = (args.image_size, args.image_size)
    
    train_loader = get_dataloader(
        args.train_images, args.train_masks,
        args.batch_size, image_size, is_train=True,
        num_workers=args.num_workers
    )
    
    val_loader = get_dataloader(
        args.val_images, args.val_masks,
        args.batch_size, image_size, is_train=False,
        num_workers=args.num_workers
    )
    
    print(f"Train set: {len(train_loader.dataset)} images")
    print(f"Val set: {len(val_loader.dataset)} images")
    
    # Créer le modèle
    print(f"\nCréation du modèle: {args.model}")
    
    if args.model == 'enet':
        model = get_enet_model(num_classes=NUM_BINARY_CLASSES)
    elif args.model == 'unet':
        model = get_unet_model(num_classes=NUM_BINARY_CLASSES, model_type='standard')
    elif args.model == 'unet_small':
        model = get_unet_model(num_classes=NUM_BINARY_CLASSES, model_type='small')
    
    model = model.to(device)
    
    # Loss function
    if args.weighted_loss:
        weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("Loss pondérée activée")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Reprendre l'entraînement
    start_epoch = 0
    if args.resume:
        print(f"\nReprise depuis {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Reprise à l'epoch {start_epoch}")
    
    # Créer le trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        experiment_name=args.experiment_name,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Entraîner
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()
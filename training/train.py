"""
Script d'entraînement principal pour la segmentation sémantique
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json
import time
from tqdm import tqdm
import argparse
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.enet import ENet
from models.unet import UNet
from data.dataset import SegmentationDataset, load_config
from data.augmentation import SegmentationTransform, ValTransform
from utils.metrics import MetricsTracker, compute_miou, compute_pixel_accuracy
from training.config import TrainingConfig, get_quick_test_config, get_full_training_config


class Trainer:
    """Classe principale pour gérer l'entraînement"""
    
    def __init__(self, config: TrainingConfig):
        """
        Args:
            config: Configuration d'entraînement
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Créer les dossiers nécessaires
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Charger la configuration des classes
        self.label_config = load_config(config.label_config)
        self.class_names = [cls['name'] for cls in self.label_config['classes']]
        
        print(f"\n{'='*60}")
        print(f"INITIALISATION DE L'ENTRAÎNEMENT")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Modèle: {config.model_name}")
        print(f"Nombre de classes: {config.num_classes}")
        print(f"Classes: {', '.join(self.class_names)}")
        print(f"{'='*60}\n")
        
        # Initialiser le modèle
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Paramètres du modèle:")
        print(f"  Total: {total_params:,}")
        print(f"  Entraînables: {trainable_params:,}\n")
        
        # Créer les datasets et dataloaders
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # Créer la loss function
        self.criterion = self._create_criterion()
        
        # Créer l'optimiseur
        self.optimizer = self._create_optimizer()
        
        # Créer le scheduler
        self.scheduler = self._create_scheduler()
        
        # Tensorboard
        if config.use_tensorboard:
            log_dir = Path(config.log_dir) / config.experiment_name
            self.writer = SummaryWriter(log_dir)
            print(f"Tensorboard log dir: {log_dir}")
        else:
            self.writer = None
        
        # Suivi de l'entraînement
        self.current_epoch = 0
        self.best_miou = 0.0
        self.best_epoch = 0
        self.train_losses = []
        self.val_mious = []
        
        # Early stopping
        self.patience_counter = 0
        
    def _create_model(self) -> nn.Module:
        """Créer le modèle"""
        if self.config.model_name.lower() == "unet":
            model = UNet(
                num_classes=self.config.num_classes,
                in_channels=3,
                **self.config.model_params
            )
        elif self.config.model_name.lower() == "enet":
            model = ENet(
                num_classes=self.config.num_classes,
                **self.config.model_params
            )
        else:
            raise ValueError(f"Modèle inconnu: {self.config.model_name}")
        
        return model
    
    def _create_dataloaders(self):
        """Créer les dataloaders pour train et validation"""
        print("Chargement des données...")
        
        # Transformations
        train_transform = SegmentationTransform(
            resize_size=self.config.resize_size,
            crop_size=self.config.crop_size,
            horizontal_flip_prob=self.config.horizontal_flip_prob,
            rotation_degrees=self.config.rotation_degrees,
            color_jitter=self.config.color_jitter,
            normalize=True
        )
        
        val_transform = ValTransform(
            resize_size=self.config.resize_size,
            normalize=True
        )
        
        # Dataset d'entraînement
        try:
            train_dataset = SegmentationDataset(
                images_dir=self.config.train_images_dir,
                masks_dir=self.config.train_masks_dir,
                transform=train_transform,
                num_classes=self.config.num_classes
            )
        except ValueError as e:
            print(f"Erreur lors du chargement du dataset d'entraînement: {e}")
            print("Assurez-vous que les dossiers contiennent des images et des masques correspondants")
            sys.exit(1)
        
        # Dataset de validation
        try:
            val_dataset = SegmentationDataset(
                images_dir=self.config.val_images_dir,
                masks_dir=self.config.val_masks_dir,
                transform=val_transform,
                num_classes=self.config.num_classes
            )
        except ValueError as e:
            print(f"Warning: Pas de dataset de validation trouvé")
            print(f"On va créer un split train/val automatique (80/20)")
            
            # Split automatique
            dataset_size = len(train_dataset)
            val_size = int(0.2 * dataset_size)
            train_size = dataset_size - val_size
            
            train_dataset, val_dataset = random_split(
                train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            # Appliquer les transformations de validation au val_dataset
            val_dataset.dataset.transform = val_transform
        
        print(f"Dataset d'entraînement: {len(train_dataset)} échantillons")
        print(f"Dataset de validation: {len(val_dataset)} échantillons\n")
        
        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        return train_loader, val_loader
    
    def _create_criterion(self) -> nn.Module:
        """Créer la fonction de loss"""
        if self.config.use_class_weights:
            # Calculer les poids des classes depuis le dataset d'entraînement
            print("Calcul des poids des classes...")
            weights = self.train_loader.dataset.get_class_weights()
            weights = weights.to(self.device)
            print(f"Poids des classes: {weights.cpu().numpy()}\n")
        else:
            weights = None
        
        criterion = nn.CrossEntropyLoss(weight=weights)

        
        return criterion
    
    def _create_optimizer(self):
        """Créer l'optimiseur"""
        if self.config.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Optimiseur inconnu: {self.config.optimizer}")
        
        return optimizer
    
    def _create_scheduler(self):
        """Créer le learning rate scheduler"""
        if not self.config.use_scheduler:
            return None
        
        if self.config.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma
            )
        elif self.config.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            raise ValueError(f"Scheduler inconnu: {self.config.scheduler_type}")
        
        return scheduler
    
    def train_epoch(self) -> float:
        """Entraîner le modèle pour une epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.num_epochs}")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculer la loss
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumuler la loss
            total_loss += loss.item()
            
            # Mettre à jour la barre de progression
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # Logger dans tensorboard
            if self.writer and batch_idx % self.config.print_every == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Valider le modèle"""
        self.model.eval()
        
        metrics_tracker = MetricsTracker(
            num_classes=self.config.num_classes,
            class_names=self.class_names
        )
        
        val_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculer la loss
            loss = self.criterion(outputs, masks)
            val_loss += loss.item()
            
            # Mettre à jour les métriques
            predictions = outputs.argmax(dim=1)
            metrics_tracker.update(predictions, masks)
        
        avg_val_loss = val_loss / len(self.val_loader)
        metrics = metrics_tracker.get_metrics()
        metrics['val_loss'] = avg_val_loss
        
        return metrics
    
    def save_checkpoint(self, metrics: dict, is_best: bool = False):
        """Sauvegarder un checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'best_miou': self.best_miou,
            'class_names': self.class_names
        }
        
        # Sauvegarder le checkpoint régulier
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint sauvegardé: {checkpoint_path}")
        
        # Sauvegarder le meilleur modèle
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"✓ Meilleur modèle sauvegardé: {best_path} (mIoU: {self.best_miou:.4f})")
        
        # Sauvegarder aussi le dernier modèle
        last_path = self.checkpoint_dir / "last_model.pth"
        torch.save(checkpoint, last_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Charger un checkpoint"""
        print(f"Chargement du checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_miou = checkpoint.get('best_miou', 0.0)
        
        print(f"Checkpoint chargé: epoch {checkpoint['epoch']}, best mIoU: {self.best_miou:.4f}")
    
    def check_early_stopping(self, current_miou: float) -> bool:
        """
        Vérifier si on doit arrêter l'entraînement
        
        Returns:
            True si on doit arrêter
        """
        if not self.config.use_early_stopping:
            return False
        
        if current_miou > self.best_miou + self.config.min_delta:
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            print(f"Early stopping counter: {self.patience_counter}/{self.config.patience}")
            
            if self.patience_counter >= self.config.patience:
                print(f"\n⚠ Early stopping déclenché après {self.config.patience} epochs sans amélioration")
                return True
        
        return False
    
    def train(self):
        """Boucle d'entraînement principale"""
        print(f"\n{'='*60}")
        print("DÉBUT DE L'ENTRAÎNEMENT")
        print(f"{'='*60}\n")
        
        # Charger un checkpoint si spécifié
        if self.config.resume_from:
            self.load_checkpoint(self.config.resume_from)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"{'='*60}")
            
            # Learning rate actuel
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")
            
            # Entraînement
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            print(f"\nTrain Loss: {train_loss:.4f}")
            
            # Validation
            if (epoch + 1) % self.config.eval_every == 0:
                print("\nÉvaluation sur le set de validation...")
                val_metrics = self.validate()
                
                current_miou = val_metrics['mIoU']
                self.val_mious.append(current_miou)
                
                print(f"\nRésultats de validation:")
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"  mIoU: {current_miou:.4f}")
                print(f"  Pixel Accuracy: {val_metrics['Pixel_Accuracy']:.2f}%")
                
                # Logger dans tensorboard
                if self.writer:
                    self.writer.add_scalar('Train/Loss', train_loss, epoch)
                    self.writer.add_scalar('Train/LR', current_lr, epoch)
                    self.writer.add_scalar('Val/Loss', val_metrics['val_loss'], epoch)
                    self.writer.add_scalar('Val/mIoU', current_miou, epoch)
                    self.writer.add_scalar('Val/PixelAccuracy', val_metrics['Pixel_Accuracy'], epoch)
                    
                    # Logger les IoU par classe
                    for cls_name in self.class_names:
                        key = f"IoU/{cls_name}"
                        if key in val_metrics:
                            self.writer.add_scalar(f'Val/IoU_{cls_name}', val_metrics[key], epoch)
                
                # Vérifier si c'est le meilleur modèle
                is_best = current_miou > self.best_miou
                if is_best:
                    improvement = current_miou - self.best_miou
                    self.best_miou = current_miou
                    self.best_epoch = epoch
                    print(f"\n✓ Nouveau meilleur mIoU: {self.best_miou:.4f} (+{improvement:.4f})")
                
                # Sauvegarder le checkpoint
                if (epoch + 1) % self.config.save_every == 0 or is_best:
                    if self.config.save_best_only and not is_best:
                        pass  # Ne sauvegarder que le meilleur
                    else:
                        self.save_checkpoint(val_metrics, is_best)
                
                # Vérifier early stopping
                if self.check_early_stopping(current_miou):
                    print(f"\nMeilleur mIoU atteint à l'epoch {self.best_epoch+1}: {self.best_miou:.4f}")
                    break
            
            # Mettre à jour le scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if (epoch + 1) % self.config.eval_every == 0:
                        self.scheduler.step(current_miou)
                else:
                    self.scheduler.step()
        
        # Fin de l'entraînement
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("ENTRAÎNEMENT TERMINÉ")
        print(f"{'='*60}")
        print(f"Temps total: {total_time/3600:.2f} heures")
        print(f"Meilleur mIoU: {self.best_miou:.4f} (epoch {self.best_epoch+1})")
        print(f"Modèles sauvegardés dans: {self.checkpoint_dir}")
        
        if self.writer:
            self.writer.close()
        
        return self.best_miou


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Entraînement de segmentation sémantique")
    parser.add_argument('--config', type=str, choices=['quick', 'full', 'high_quality'], 
                       default='full', help='Configuration prédéfinie')
    parser.add_argument('--train_images', type=str, help='Dossier des images d\'entraînement')
    parser.add_argument('--train_masks', type=str, help='Dossier des masques d\'entraînement')
    parser.add_argument('--val_images', type=str, help='Dossier des images de validation')
    parser.add_argument('--val_masks', type=str, help='Dossier des masques de validation')
    parser.add_argument('--model', type=str, choices=['unet', 'enet'], 
                       default='unet', help='Architecture du modèle')
    parser.add_argument('--batch_size', type=int, help='Taille du batch')
    parser.add_argument('--epochs', type=int, help='Nombre d\'epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--resume', type=str, help='Checkpoint à reprendre')
    parser.add_argument('--experiment_name', type=str, default='segmentation_exp',
                       help='Nom de l\'expérience')
    
    args = parser.parse_args()
    
    # Charger la configuration de base
    if args.config == 'quick':
        config = get_quick_test_config()
    elif args.config == 'high_quality':
        config = get_high_quality_config()
    else:
        config = get_full_training_config()
    
    # Override avec les arguments en ligne de commande
    if args.train_images:
        config.train_images_dir = args.train_images
    if args.train_masks:
        config.train_masks_dir = args.train_masks
    if args.val_images:
        config.val_images_dir = args.val_images
    if args.val_masks:
        config.val_masks_dir = args.val_masks
    if args.model:
        config.model_name = args.model
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.resume:
        config.resume_from = args.resume
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    # Créer le trainer et lancer l'entraînement
    trainer = Trainer(config)
    best_miou = trainer.train()
    
    print(f"\n✓ Entraînement terminé avec succès!")
    print(f"Meilleur mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
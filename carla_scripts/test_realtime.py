"""
Test du modèle en temps réel dans CARLA
Affiche la segmentation binaire (Traversable/Obstrué) en temps réel
"""

import glob
import os
import sys
import argparse
import time
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
from torchvision import transforms

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Ajouter le chemin parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.enet import get_enet_model
from models.unet import get_unet_model
from config import BINARY_COLORS, BINARY_CLASSES


class RealtimeSegmentation:
    """Segmentation en temps réel dans CARLA"""
    
    def __init__(self, checkpoint_path, model_type='enet', device='cuda', image_size=(512, 512)):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        print(f"Device: {self.device}")
        
        # Charger le modèle
        print(f"Chargement du modèle depuis {checkpoint_path}...")
        self.model = self._load_model(checkpoint_path, model_type)
        self.model.eval()
        
        # Transformation
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Buffers
        self.current_image = None
        self.current_prediction = None
        
        # Stats
        self.frame_count = 0
        self.total_time = 0.0
        
        print("Modèle chargé avec succès!")
    
    def _load_model(self, checkpoint_path, model_type):
        """Charge le modèle depuis un checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Créer le modèle
        if model_type == 'enet':
            model = get_enet_model(num_classes=2)
        elif model_type == 'unet':
            model = get_unet_model(num_classes=2, model_type='standard')
        elif model_type == 'unet_small':
            model = get_unet_model(num_classes=2, model_type='small')
        else:
            raise ValueError(f"Type de modèle inconnu: {model_type}")
        
        # Charger les poids
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def setup_carla(self, host='localhost', port=2000):
        """Configure la connexion CARLA"""
        print(f"Connexion à CARLA ({host}:{port})...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        
        print(f"CARLA connecté. Carte: {self.world.get_map().name}")
    
    def spawn_vehicle_with_camera(self, camera_width=800, camera_height=600):
        """Spawn un véhicule avec caméra"""
        # Spawn véhicule
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = np.random.choice(spawn_points)
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Véhicule spawné à {spawn_point.location}")
        
        # Caméra RGB
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(camera_width))
        camera_bp.set_attribute('image_size_y', str(camera_height))
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle)
        
        # Callback
        self.camera.listen(lambda image: self._process_image(image))
        
        # Activer autopilot
        self.vehicle.set_autopilot(True)
        
        print("Caméra configurée et autopilot activé")
    
    def _process_image(self, image):
        """Traite l'image de la caméra"""
        # Convertir en numpy
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # RGB
        
        self.current_image = array
    
    @torch.no_grad()
    def predict(self, image):
        """
        Prédit la segmentation pour une image
        
        Args:
            image: Image RGB (H, W, 3)
        
        Returns:
            prediction: Masque de prédiction (H, W)
        """
        # Convertir en PIL
        pil_image = Image.fromarray(image)
        
        # Appliquer les transformations
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Prédiction
        start_time = time.time()
        output = self.model(input_tensor)
        inference_time = time.time() - start_time
        
        # Argmax pour obtenir les classes
        prediction = torch.argmax(output, dim=1).squeeze(0)
        
        # Redimensionner à la taille originale
        prediction = F.interpolate(
            prediction.unsqueeze(0).unsqueeze(0).float(),
            size=(image.shape[0], image.shape[1]),
            mode='nearest'
        ).squeeze().long()
        
        prediction = prediction.cpu().numpy()
        
        # Stats
        self.frame_count += 1
        self.total_time += inference_time
        
        return prediction, inference_time
    
    def colorize_prediction(self, prediction):
        """
        Colorise une prédiction binaire
        
        Args:
            prediction: Masque (H, W)
        
        Returns:
            colored: Image RGB colorée
        """
        h, w = prediction.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_idx, color in BINARY_COLORS.items():
            colored[prediction == class_idx] = color
        
        return colored
    
    def create_overlay(self, image, prediction, alpha=0.5):
        """
        Crée une image avec overlay de la prédiction
        
        Args:
            image: Image originale
            prediction: Masque de prédiction
            alpha: Transparence
        
        Returns:
            overlay: Image avec overlay
        """
        colored_pred = self.colorize_prediction(prediction)
        overlay = cv2.addWeighted(image, 1 - alpha, colored_pred, alpha, 0)
        return overlay
    
    def add_legend(self, image):
        """Ajoute une légende à l'image"""
        h, w = image.shape[:2]
        legend_h = 60
        
        # Créer une zone de légende
        legend = np.ones((legend_h, w, 3), dtype=np.uint8) * 255
        
        # Traversable
        cv2.rectangle(legend, (10, 15), (40, 45), BINARY_COLORS[0], -1)
        cv2.putText(legend, 'Traversable', (50, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Obstrué
        cv2.rectangle(legend, (200, 15), (230, 45), BINARY_COLORS[1], -1)
        cv2.putText(legend, 'Obstructed', (240, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Concaténer
        result = np.vstack([image, legend])
        
        return result
    
    def run(self, display_width=1280, display_height=720, save_video=False, 
            video_path='output.avi', show_overlay=True):
        """
        Lance la boucle principale
        
        Args:
            display_width: Largeur de la fenêtre d'affichage
            display_height: Hauteur de la fenêtre d'affichage
            save_video: Sauvegarder la vidéo
            video_path: Chemin de la vidéo
            show_overlay: Afficher l'overlay ou seulement la prédiction
        """
        print("\n" + "="*60)
        print("SEGMENTATION EN TEMPS RÉEL")
        print("="*60)
        print("Appuyez sur 'q' pour quitter")
        print("Appuyez sur 's' pour sauvegarder la frame actuelle")
        print("="*60 + "\n")
        
        # Initialiser le writer vidéo si nécessaire
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(video_path, fourcc, 20.0,
                                          (display_width, display_height))
            print(f"Enregistrement vidéo: {video_path}")
        
        try:
            while True:
                if self.current_image is not None:
                    # Prédiction
                    prediction, inference_time = self.predict(self.current_image)
                    
                    # Créer la visualisation
                    if show_overlay:
                        display = self.create_overlay(self.current_image, prediction)
                    else:
                        display = self.colorize_prediction(prediction)
                    
                    # Ajouter la légende
                    display = self.add_legend(display)
                    
                    # Ajouter les stats
                    fps = self.frame_count / self.total_time if self.total_time > 0 else 0
                    cv2.putText(display, f'FPS: {fps:.1f}', (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(display, f'Inference: {inference_time*1000:.1f}ms', (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Redimensionner pour l'affichage
                    display_resized = cv2.resize(display, (display_width, display_height))
                    
                    # Afficher
                    cv2.imshow('Segmentation Temps Réel', display_resized)
                    
                    # Sauvegarder dans la vidéo
                    if video_writer:
                        video_writer.write(display_resized)
                    
                    # Gestion des touches
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        filename = f'frame_{self.frame_count:06d}.png'
                        cv2.imwrite(filename, display)
                        print(f"Frame sauvegardée: {filename}")
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\nInterruption par l'utilisateur")
        finally:
            self.cleanup(video_writer)
    
    def cleanup(self, video_writer=None):
        """Nettoie les ressources"""
        print("\nNettoyage...")
        
        # Stats finales
        if self.frame_count > 0:
            avg_fps = self.frame_count / self.total_time
            avg_time = (self.total_time / self.frame_count) * 1000
            
            print(f"\n{'='*60}")
            print("STATISTIQUES")
            print(f"{'='*60}")
            print(f"Frames traitées: {self.frame_count}")
            print(f"FPS moyen: {avg_fps:.2f}")
            print(f"Temps d'inférence moyen: {avg_time:.2f}ms")
            print(f"{'='*60}\n")
        
        # Fermer la vidéo
        if video_writer:
            video_writer.release()
            print("Vidéo sauvegardée")
        
        # Fermer OpenCV
        cv2.destroyAllWindows()
        
        # Nettoyer CARLA
        if hasattr(self, 'camera'):
            self.camera.stop()
            self.camera.destroy()
        if hasattr(self, 'vehicle'):
            self.vehicle.destroy()
        
        print("Nettoyage terminé")


def main():
    parser = argparse.ArgumentParser(description='Test temps réel dans CARLA')
    
    # Modèle
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Chemin vers le checkpoint du modèle')
    parser.add_argument('--model', type=str, default='enet',
                       choices=['enet', 'unet', 'unet_small'],
                       help='Type de modèle')
    
    # CARLA
    parser.add_argument('--host', type=str, default='localhost',
                       help='Hôte CARLA')
    parser.add_argument('--port', type=int, default=2000,
                       help='Port CARLA')
    parser.add_argument('--camera_width', type=int, default=800,
                       help='Largeur de la caméra')
    parser.add_argument('--camera_height', type=int, default=600,
                       help='Hauteur de la caméra')
    
    # Affichage
    parser.add_argument('--display_width', type=int, default=1280,
                       help='Largeur de la fenêtre')
    parser.add_argument('--display_height', type=int, default=720,
                       help='Hauteur de la fenêtre')
    parser.add_argument('--no_overlay', action='store_true',
                       help='Afficher seulement la prédiction')
    
    # Vidéo
    parser.add_argument('--save_video', action='store_true',
                       help='Sauvegarder une vidéo')
    parser.add_argument('--video_path', type=str, default='output.avi',
                       help='Chemin de la vidéo de sortie')
    
    # Modèle
    parser.add_argument('--image_size', type=int, default=512,
                       help='Taille d\'entrée du modèle')
    
    args = parser.parse_args()
    
    # Créer le système de segmentation
    segmentation = RealtimeSegmentation(
        checkpoint_path=args.checkpoint,
        model_type=args.model,
        device='cuda',
        image_size=(args.image_size, args.image_size)
    )
    
    # Configurer CARLA
    segmentation.setup_carla(host=args.host, port=args.port)
    segmentation.spawn_vehicle_with_camera(
        camera_width=args.camera_width,
        camera_height=args.camera_height
    )
    
    # Attendre que la caméra soit prête
    print("Attente de la première frame...")
    while segmentation.current_image is None:
        time.sleep(0.1)
    
    print("Démarrage de la segmentation...\n")
    
    # Lancer la boucle principale
    segmentation.run(
        display_width=args.display_width,
        display_height=args.display_height,
        save_video=args.save_video,
        video_path=args.video_path,
        show_overlay=not args.no_overlay
    )


if __name__ == '__main__':
    main()
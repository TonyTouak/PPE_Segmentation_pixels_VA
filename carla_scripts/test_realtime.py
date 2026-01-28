"""
Test du modèle entraîné en temps réel dans CARLA
"""

import carla
import numpy as np
import cv2
import torch
from pathlib import Path
import time
import argparse
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.enet import ENet
from models.unet import UNet
from data.dataset import load_config


class RealtimeSegmentation:
    """Test de segmentation en temps réel dans CARLA"""
    
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Args:
            checkpoint_path: Chemin vers le checkpoint du modèle
            device: Device à utiliser
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        print(f"Device: {self.device}")
        print(f"Chargement du modèle: {checkpoint_path}")
        
        # Charger le checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint['config']
        self.class_names = checkpoint['class_names']
        
        # Charger la configuration des couleurs
        self.class_names = ["obstructed", "traversable"]
        self.class_colors = {
            0: (255, 0, 0),     # rouge = obstrué
            1: (0, 255, 0)      # vert = traversable
        }

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
        
        print(f"✓ Modèle chargé: {self.config.model_name}")
        print(f"✓ Epoch: {checkpoint['epoch']}, Best mIoU: {checkpoint.get('best_miou', 'N/A')}")
        
        # Normalisation ImageNet
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        
        # CARLA
        print("\nConnexion à CARLA...")
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Mode synchrone
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
        self.blueprint_library = self.world.get_blueprint_library()
        
        self.vehicle = None
        self.camera = None
        self.current_image = None
        
        # Statistiques
        self.fps_history = []
        self.inference_times = []
        
        print("✓ Connecté à CARLA")
    
    def spawn_vehicle(self):
        """Spawner un véhicule"""
        print("Spawn du véhicule...")
        
        import random
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(True)
        
        print(f"✓ Véhicule spawné")
        time.sleep(2)
    
    def setup_camera(self, width: int = 800, height: int = 600):
        """
        Configurer la caméra
        
        Args:
            width: Largeur de l'image
            height: Hauteur de l'image
        """
        print(f"Configuration de la caméra ({width}x{height})...")
        
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(width))
        camera_bp.set_attribute('image_size_y', str(height))
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=-15)
        )
        
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )
        
        self.camera.listen(lambda image: self._on_image(image))
        
        print("✓ Caméra configurée")
    
    def _on_image(self, image):
        """Callback pour les images"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # RGB
        self.current_image = array
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Prétraiter l'image pour le modèle
        
        Args:
            image: Image RGB numpy array
            
        Returns:
            Tensor prétraité
        """
        # Resize
        image = cv2.resize(image, self.config.resize_size, interpolation=cv2.INTER_LINEAR)
        
        # Normaliser
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        # Convertir en tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image = image.unsqueeze(0)  # Ajouter dimension batch
        
        return image
    
    @torch.no_grad()
    def predict(self, image: np.ndarray) -> tuple:
        """
        Faire une prédiction
        
        Args:
            image: Image RGB numpy array
            
        Returns:
            (prediction_mask, colored_mask, inference_time)
        """
        start_time = time.time()
        
        # Prétraiter
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # Prédiction
        output = self.model(input_tensor)
        prediction = output.argmax(dim=1).squeeze(0).cpu().numpy()
        
        inference_time = time.time() - start_time
        
        # Redimensionner à la taille originale
        original_h, original_w = image.shape[:2]
        prediction = cv2.resize(
            prediction.astype(np.uint8),
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Convertir en couleur
        colored_mask = self.class_colors[prediction]
        
        return prediction, colored_mask, inference_time
    
    def create_visualization(self, image: np.ndarray, colored_mask: np.ndarray,
                           inference_time: float, fps: float) -> np.ndarray:
        """
        Créer une visualisation combinée
        
        Args:
            image: Image originale
            colored_mask: Masque coloré
            inference_time: Temps d'inférence
            fps: FPS actuel
            
        Returns:
            Image de visualisation
        """
        h, w = image.shape[:2]
        
        # Créer l'overlay
        overlay = cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            0.6,
            cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR),
            0.4,
            0
        )
        
        # Créer une visualisation côte à côte
        vis = np.zeros((h, w * 2, 3), dtype=np.uint8)
        vis[:, :w] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        vis[:, w:] = overlay
        
        # Ajouter les informations
        info_lines = [
            f"Original",
            f"FPS: {fps:.1f}",
            f"Inference: {inference_time*1000:.1f}ms"
        ]
        
        y_offset = 30
        for line in info_lines:
            cv2.putText(vis, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            y_offset += 30
        
        cv2.putText(vis, "Segmentation", (w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, "Segmentation", (w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Ajouter la légende des classes
        legend_y = h - 30 * min(len(self.class_names), 5)
        for i, name in enumerate(self.class_names[:5]):  # Limiter à 5 classes
            color = tuple(int(c) for c in self.class_colors[i])
            cv2.rectangle(vis, (w + 10, legend_y), (w + 30, legend_y + 20), 
                         color[::-1], -1)  # BGR
            cv2.putText(vis, name, (w + 35, legend_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(vis, name, (w + 35, legend_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            legend_y += 30
        
        return vis
    
    def run(self, duration: float = None, save_video: bool = False, 
            output_path: str = "output_video.avi"):
        """
        Lancer le test en temps réel
        
        Args:
            duration: Durée du test en secondes (None = infini)
            save_video: Sauvegarder la vidéo
            output_path: Chemin de sortie pour la vidéo
        """
        print("\n" + "="*60)
        print("DÉMARRAGE DU TEST EN TEMPS RÉEL")
        print("="*60)
        print("Appuyez sur 'q' pour quitter")
        print("Appuyez sur 's' pour sauvegarder une capture")
        print("="*60 + "\n")
        
        # Créer la fenêtre OpenCV
        cv2.namedWindow('CARLA Segmentation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('CARLA Segmentation', 1600, 600)
        
        # Initialiser l'enregistrement vidéo si demandé
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (1600, 600))
            print(f"Enregistrement vidéo: {output_path}")
        
        start_time = time.time()
        frame_count = 0
        capture_count = 0
        
        try:
            while True:
                # Tick de la simulation
                self.world.tick()
                
                # Vérifier si on a une image
                if self.current_image is None:
                    continue
                
                # Faire la prédiction
                frame_start = time.time()
                prediction, colored_mask, inference_time = self.predict(self.current_image)
                
                # Calculer FPS
                frame_time = time.time() - frame_start
                fps = 1.0 / frame_time if frame_time > 0 else 0
                self.fps_history.append(fps)
                self.inference_times.append(inference_time)
                
                # Créer la visualisation
                vis = self.create_visualization(
                    self.current_image,
                    colored_mask,
                    inference_time,
                    fps
                )
                
                # Afficher
                cv2.imshow('CARLA Segmentation', vis)
                
                # Enregistrer la vidéo si demandé
                if video_writer:
                    video_writer.write(vis)
                
                frame_count += 1
                
                # Gestion des touches
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nArrêt demandé par l'utilisateur")
                    break
                elif key == ord('s'):
                    # Sauvegarder une capture
                    capture_path = f"capture_{capture_count:04d}.png"
                    cv2.imwrite(capture_path, vis)
                    print(f"\n✓ Capture sauvegardée: {capture_path}")
                    capture_count += 1
                
                # Vérifier la durée
                if duration and (time.time() - start_time) > duration:
                    print(f"\nDurée de {duration}s atteinte")
                    break
        
        except KeyboardInterrupt:
            print("\n\nInterruption par l'utilisateur")
        
        finally:
            # Fermer la vidéo
            if video_writer:
                video_writer.release()
                print(f"✓ Vidéo sauvegardée: {output_path}")
            
            cv2.destroyAllWindows()
            
            # Afficher les statistiques
            if len(self.fps_history) > 0:
                print("\n" + "="*60)
                print("STATISTIQUES")
                print("="*60)
                print(f"Frames traitées: {frame_count}")
                print(f"Durée totale: {time.time() - start_time:.2f}s")
                print(f"FPS moyen: {np.mean(self.fps_history):.2f}")
                print(f"FPS min: {np.min(self.fps_history):.2f}")
                print(f"FPS max: {np.max(self.fps_history):.2f}")
                print(f"Temps d'inférence moyen: {np.mean(self.inference_times)*1000:.2f}ms")
                print(f"Temps d'inférence min: {np.min(self.inference_times)*1000:.2f}ms")
                print(f"Temps d'inférence max: {np.max(self.inference_times)*1000:.2f}ms")
                print("="*60)
    
    def cleanup(self):
        """Nettoyer les ressources"""
        print("\nNettoyage...")
        
        if self.camera:
            self.camera.stop()
            self.camera.destroy()
        
        if self.vehicle:
            self.vehicle.destroy()
        
        # Restaurer les paramètres
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        
        print("✓ Nettoyage terminé")


def main():
    parser = argparse.ArgumentParser(description="Test en temps réel dans CARLA")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Chemin vers le checkpoint du modèle')
    parser.add_argument('--device', type=str, default=None,
                       help='Device à utiliser (cuda/cpu)')
    parser.add_argument('--width', type=int, default=800,
                       help='Largeur des images')
    parser.add_argument('--height', type=int, default=600,
                       help='Hauteur des images')
    parser.add_argument('--duration', type=float, default=None,
                       help='Durée du test en secondes')
    parser.add_argument('--save_video', action='store_true',
                       help='Sauvegarder la vidéo')
    parser.add_argument('--output', type=str, default='output_video.avi',
                       help='Chemin de sortie pour la vidéo')
    
    args = parser.parse_args()
    
    # Créer le testeur
    tester = RealtimeSegmentation(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    try:
        tester.spawn_vehicle()
        tester.setup_camera(width=args.width, height=args.height)
        tester.run(
            duration=args.duration,
            save_video=args.save_video,
            output_path=args.output
        )
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
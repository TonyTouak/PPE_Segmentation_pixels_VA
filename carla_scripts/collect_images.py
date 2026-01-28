"""
Collecte d'images depuis CARLA pour l'annotation
"""

import carla
import numpy as np
import cv2
from pathlib import Path
import time
import random
import argparse


class CarlaImageCollector:
    """Collecteur d'images depuis CARLA"""
    
    def __init__(self, output_dir: str = 'data/collected_images', 
                 image_width: int = 800, image_height: int = 600):
        """
        Args:
            output_dir: Dossier de sortie
            image_width: Largeur des images
            image_height: Hauteur des images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_width = image_width
        self.image_height = image_height
        
        # Connexion à CARLA
        print("Connexion à CARLA...")
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Paramètres de simulation
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
        self.blueprint_library = self.world.get_blueprint_library()
        
        self.vehicle = None
        self.camera = None
        self.semantic_camera = None
        
        self.image_queue = []
        self.semantic_queue = []
        self.frame_count = 0
        
        print(f"✓ Connecté à CARLA")
        print(f"Images seront sauvegardées dans: {self.output_dir}")
    
    def spawn_vehicle(self):
        """Spawner un véhicule"""
        print("Spawn du véhicule...")
        
        # Trouver un spawn point
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        # Créer le véhicule
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # Activer l'autopilot
        self.vehicle.set_autopilot(True)
        
        print(f"✓ Véhicule spawné: {self.vehicle.type_id}")
        
        time.sleep(2)  # Attendre que le véhicule se stabilise
    
    def setup_cameras(self):
        """Configurer les caméras"""
        print("Configuration des caméras...")
        
        # Caméra RGB
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.image_width))
        camera_bp.set_attribute('image_size_y', str(self.image_height))
        camera_bp.set_attribute('fov', '90')
        
        # Caméra sémantique (pour référence/test)
        semantic_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        semantic_bp.set_attribute('image_size_x', str(self.image_width))
        semantic_bp.set_attribute('image_size_y', str(self.image_height))
        semantic_bp.set_attribute('fov', '90')
        
        # Position de la caméra
        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=-15)
        )
        
        # Spawner les caméras
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )
        
        self.semantic_camera = self.world.spawn_actor(
            semantic_bp,
            camera_transform,
            attach_to=self.vehicle
        )
        
        # Callbacks
        self.camera.listen(lambda image: self._on_rgb_image(image))
        self.semantic_camera.listen(lambda image: self._on_semantic_image(image))
        
        print("✓ Caméras configurées")
    
    def _on_rgb_image(self, image):
        """Callback pour les images RGB"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((self.image_height, self.image_width, 4))
        array = array[:, :, :3]  # Enlever le canal alpha
        self.image_queue.append((image.frame, array))
    
    def _on_semantic_image(self, image):
        """Callback pour les images sémantiques"""
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((self.image_height, self.image_width, 4))
        array = array[:, :, 2]  # Canal R contient les labels
        self.semantic_queue.append((image.frame, array))
    
    def collect(self, num_images: int = 100, interval: int = 10):
        """
        Collecter des images
        
        Args:
            num_images: Nombre d'images à collecter
            interval: Intervalle entre les captures (en frames)
        """
        print(f"\nCollecte de {num_images} images (intervalle: {interval} frames)...")
        
        collected = 0
        frame_counter = 0
        
        try:
            while collected < num_images:
                # Tick de la simulation
                self.world.tick()
                frame_counter += 1
                
                # Capturer à intervalles réguliers
                if frame_counter % interval == 0 and len(self.image_queue) > 0:
                    # Récupérer les images
                    frame_rgb, rgb_image = self.image_queue.pop(0)
                    
                    # Trouver l'image sémantique correspondante
                    semantic_image = None
                    for i, (frame_sem, sem_img) in enumerate(self.semantic_queue):
                        if frame_sem == frame_rgb:
                            semantic_image = sem_img
                            self.semantic_queue.pop(i)
                            break
                    
                    # Sauvegarder
                    if semantic_image is not None:
                        rgb_path = self.output_dir / f"image_{self.frame_count:06d}.png"
                        sem_path = self.output_dir / f"semantic_{self.frame_count:06d}.png"
                        
                        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(str(sem_path), semantic_image)
                        
                        collected += 1
                        self.frame_count += 1
                        
                        print(f"✓ Collecté: {collected}/{num_images} - Frame: {frame_rgb}", end='\r')
        
        except KeyboardInterrupt:
            print("\n\nCollecte interrompue par l'utilisateur")
        
        print(f"\n\n✓ Collecte terminée: {collected} images sauvegardées")
    
    def cleanup(self):
        """Nettoyer les ressources"""
        print("\nNettoyage...")
        
        if self.camera:
            self.camera.stop()
            self.camera.destroy()
        
        if self.semantic_camera:
            self.semantic_camera.stop()
            self.semantic_camera.destroy()
        
        if self.vehicle:
            self.vehicle.destroy()
        
        # Restaurer les paramètres
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        
        print("✓ Nettoyage terminé")


def main():
    parser = argparse.ArgumentParser(description="Collecte d'images depuis CARLA")
    parser.add_argument('--output', type=str, default='data/collected_images',
                       help='Dossier de sortie')
    parser.add_argument('--num_images', type=int, default=100,
                       help='Nombre d\'images à collecter')
    parser.add_argument('--interval', type=int, default=10,
                       help='Intervalle entre les captures (en frames)')
    parser.add_argument('--width', type=int, default=800,
                       help='Largeur des images')
    parser.add_argument('--height', type=int, default=600,
                       help='Hauteur des images')
    
    args = parser.parse_args()
    
    collector = CarlaImageCollector(
        output_dir=args.output,
        image_width=args.width,
        image_height=args.height
    )
    
    try:
        collector.spawn_vehicle()
        collector.setup_cameras()
        collector.collect(num_images=args.num_images, interval=args.interval)
    finally:
        collector.cleanup()


if __name__ == "__main__":
    main()
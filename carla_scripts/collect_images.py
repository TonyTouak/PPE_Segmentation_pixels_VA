"""
Script de collecte d'images RGB et masques de segmentation depuis CARLA
Collecte automatique dans différentes conditions et environnements
"""

import glob
import os
import sys
import argparse
import time
import numpy as np
from PIL import Image

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


class DataCollector:
    """Collecteur de données depuis CARLA"""
    
    def __init__(self, output_dir, image_size=(800, 600)):
        self.output_dir = output_dir
        self.image_size = image_size
        
        # Créer les dossiers
        self.rgb_dir = os.path.join(output_dir, 'images')
        self.seg_dir = os.path.join(output_dir, 'masks')
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.seg_dir, exist_ok=True)
        
        # Compteur d'images
        self.image_count = 0
        
        # Buffers pour les images
        self.rgb_image = None
        self.seg_image = None
        
        print(f"DataCollector initialisé. Sortie: {output_dir}")
    
    def setup_carla(self, host='localhost', port=2000):
        """Connexion à CARLA et configuration"""
        print(f"Connexion à CARLA ({host}:{port})...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        
        print(f"CARLA connecté. Carte: {self.world.get_map().name}")
    
    def spawn_vehicle_with_cameras(self):
        """Spawn un véhicule avec caméras RGB et sémantique"""
        # Spawn du véhicule
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = np.random.choice(spawn_points)
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Véhicule spawné à {spawn_point.location}")
        
        # Configuration caméra RGB
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.image_size[0]))
        camera_bp.set_attribute('image_size_y', str(self.image_size[1]))
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera_rgb = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle)
        
        # Configuration caméra sémantique
        sem_camera_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        sem_camera_bp.set_attribute('image_size_x', str(self.image_size[0]))
        sem_camera_bp.set_attribute('image_size_y', str(self.image_size[1]))
        sem_camera_bp.set_attribute('fov', '90')
        
        self.camera_sem = self.world.spawn_actor(
            sem_camera_bp, camera_transform, attach_to=self.vehicle)
        
        # Callbacks
        self.camera_rgb.listen(lambda image: self._process_rgb(image))
        self.camera_sem.listen(lambda image: self._process_semantic(image))
        
        print("Caméras configurées")
    
    def _process_rgb(self, image):
        """Traite l'image RGB"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Enlever le canal alpha
        self.rgb_image = array
    
    def _process_semantic(self, image):
        """Traite l'image de segmentation sémantique"""
        # CARLA renvoie les classes encodées en rouge
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        # Le canal rouge contient la classe
        semantic_data = array[:, :, 2]  # Canal rouge = classe sémantique
        self.seg_image = semantic_data
    
    def save_current_frame(self):
        """Sauvegarde la frame actuelle si les deux images sont prêtes"""
        if self.rgb_image is not None and self.seg_image is not None:
            filename = f"frame_{self.image_count:06d}"
            
            # Sauvegarder RGB
            rgb_path = os.path.join(self.rgb_dir, f"{filename}.png")
            Image.fromarray(self.rgb_image).save(rgb_path)
            
            # Sauvegarder masque sémantique (format numpy pour préserver les valeurs exactes)
            seg_path = os.path.join(self.seg_dir, f"{filename}.npy")
            np.save(seg_path, self.seg_image)
            
            # Optionnel: sauvegarder aussi en PNG pour visualisation
            seg_path_png = os.path.join(self.seg_dir, f"{filename}.png")
            Image.fromarray(self.seg_image).save(seg_path_png)
            
            self.image_count += 1
            
            if self.image_count % 10 == 0:
                print(f"Images collectées: {self.image_count}")
            
            return True
        return False
    
    def set_weather(self, weather_preset):
        """Change les conditions météo"""
        weather_presets = {
            'clear': carla.WeatherParameters.ClearNoon,
            'cloudy': carla.WeatherParameters.CloudyNoon,
            'wet': carla.WeatherParameters.WetNoon,
            'rain': carla.WeatherParameters.HardRainNoon,
            'fog': carla.WeatherParameters.SoftRainSunset,
            'night': carla.WeatherParameters.ClearNight,
        }
        
        if weather_preset in weather_presets:
            self.world.set_weather(weather_presets[weather_preset])
            print(f"Météo changée: {weather_preset}")
    
    def enable_autopilot(self):
        """Active le pilote automatique"""
        self.vehicle.set_autopilot(True)
        print("Autopilot activé")
    
    def collect_diverse_data(self, num_images_per_weather=100):
        """
        Collecte des données dans différentes conditions
        """
        weather_conditions = ['clear', 'cloudy', 'wet', 'rain', 'fog']
        
        for weather in weather_conditions:
            print(f"\n=== Collecte avec météo: {weather} ===")
            self.set_weather(weather)
            time.sleep(2)  # Attendre que la météo s'applique
            
            collected = 0
            while collected < num_images_per_weather:
                if self.save_current_frame():
                    collected += 1
                time.sleep(0.1)  # 10 FPS
            
            print(f"Collecté {collected} images pour {weather}")
    
    def cleanup(self):
        """Nettoie les ressources"""
        print("\nNettoyage...")
        if hasattr(self, 'camera_rgb'):
            self.camera_rgb.stop()
            self.camera_rgb.destroy()
        if hasattr(self, 'camera_sem'):
            self.camera_sem.stop()
            self.camera_sem.destroy()
        if hasattr(self, 'vehicle'):
            self.vehicle.destroy()
        print("Nettoyage terminé")


def main():
    parser = argparse.ArgumentParser(description='Collecte de données depuis CARLA')
    parser.add_argument('--output', type=str, default='data/collected',
                       help='Dossier de sortie')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Hôte CARLA')
    parser.add_argument('--port', type=int, default=2000,
                       help='Port CARLA')
    parser.add_argument('--num_images', type=int, default=500,
                       help='Nombre d\'images à collecter')
    parser.add_argument('--diverse', action='store_true',
                       help='Collecter dans différentes conditions météo')
    parser.add_argument('--width', type=int, default=800,
                       help='Largeur des images')
    parser.add_argument('--height', type=int, default=600,
                       help='Hauteur des images')
    
    args = parser.parse_args()
    
    # Créer le collecteur
    collector = DataCollector(
        output_dir=args.output,
        image_size=(args.width, args.height)
    )
    
    try:
        # Connexion à CARLA
        collector.setup_carla(host=args.host, port=args.port)
        
        # Spawn véhicule avec caméras
        collector.spawn_vehicle_with_cameras()
        
        # Activer autopilot
        collector.enable_autopilot()
        
        # Collecter les données
        if args.diverse:
            images_per_weather = args.num_images // 5
            collector.collect_diverse_data(num_images_per_weather=images_per_weather)
        else:
            print(f"\n=== Collecte de {args.num_images} images ===")
            collected = 0
            while collected < args.num_images:
                if collector.save_current_frame():
                    collected += 1
                time.sleep(0.1)
        
        print(f"\n✓ Collecte terminée! {collector.image_count} images sauvegardées")
        print(f"  RGB: {collector.rgb_dir}")
        print(f"  Masks: {collector.seg_dir}")
        
    except KeyboardInterrupt:
        print("\nInterruption par l'utilisateur")
    except Exception as e:
        print(f"\nErreur: {e}")
        import traceback
        traceback.print_exc()
    finally:
        collector.cleanup()


if __name__ == '__main__':
    main()
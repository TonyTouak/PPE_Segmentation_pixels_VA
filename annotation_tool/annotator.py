"""
Outil d'annotation manuelle pour la segmentation sémantique
Interface interactive pour annoter les images pixel par pixel
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import argparse

class SemanticAnnotator:
    def __init__(self, images_dir: str, output_dir: str, config_file: str):
        """
        Outil d'annotation interactif
        
        Args:
            images_dir: Dossier contenant les images à annoter
            output_dir: Dossier de sortie pour les masques
            config_file: Fichier JSON avec les classes et couleurs
        """
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.traversable_map = {
            cls['id']: cls.get('traversable', False)
            for cls in self.classes
        }

        
        # Charger la configuration des classes
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        self.classes = config['classes']
        self.colors = {cls['id']: tuple(cls['color']) for cls in self.classes}
        self.class_names = {cls['id']: cls['name'] for cls in self.classes}
        
        # État de l'annotation
        self.current_class = 0
        self.brush_size = 10
        self.drawing = False
        self.image = None
        self.mask = None
        self.overlay = None
        self.images = sorted(list(self.images_dir.glob('*.png')) + 
                           list(self.images_dir.glob('*.jpg')))
        self.current_image_idx = 0
        
        # Historique pour undo
        self.history = []
        self.max_history = 20
        
        print(f"Images trouvées: {len(self.images)}")
        print("\n=== CONTRÔLES ===")
        print("Souris gauche: Dessiner")
        print("Souris droite: Effacer")
        print("Molette haut/bas: Changer taille du pinceau")
        print("1-9: Sélectionner la classe")
        print("n: Image suivante")
        print("p: Image précédente")
        print("s: Sauvegarder")
        print("z: Undo")
        print("c: Effacer tout")
        print("t: Toggle overlay (afficher/cacher l'annotation)")
        print("q: Quitter")
        print("=================\n")


    def generate_binary_mask(self):
        """
        Génère un masque binaire :
        1 = traversable
        0 = obstrué
        """
        binary_mask = np.zeros_like(self.mask, dtype=np.uint8)

        for class_id, is_traversable in self.traversable_map.items():
            if is_traversable:
                binary_mask[self.mask == class_id] = 1

        return binary_mask

        
    def load_image(self, idx: int):
        """Charger une image et son masque s'il existe"""
        if idx < 0 or idx >= len(self.images):
            return
        
        self.current_image_idx = idx
        image_path = self.images[idx]
        
        # Charger l'image
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            print(f"Erreur lors du chargement de {image_path}")
            return
        
        # Charger ou créer le masque
        mask_path = self.output_dir / f"{image_path.stem}_mask.png"
        if mask_path.exists():
            self.mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            print(f"Masque existant chargé: {mask_path.name}")
        else:
            self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            print(f"Nouveau masque créé pour: {image_path.name}")
        
        # Créer l'overlay
        self.update_overlay()
        self.history = []
        
        print(f"\nImage {idx + 1}/{len(self.images)}: {image_path.name}")
        print(f"Classe actuelle: {self.class_names[self.current_class]} (ID: {self.current_class})")
        
    def update_overlay(self):
        """Mettre à jour l'overlay de visualisation"""
        self.overlay = self.image.copy()
        
        # Créer une image colorée du masque
        color_mask = np.zeros_like(self.image)
        for class_id, color in self.colors.items():
            color_mask[self.mask == class_id] = color
        
        # Mélanger avec l'image originale
        self.overlay = cv2.addWeighted(self.image, 0.6, color_mask, 0.4, 0)
        
    def save_history(self):
        """Sauvegarder l'état actuel dans l'historique"""
        if len(self.history) >= self.max_history:
            self.history.pop(0)
        self.history.append(self.mask.copy())
        
    def undo(self):
        """Annuler la dernière action"""
        if self.history:
            self.mask = self.history.pop()
            self.update_overlay()
            print("Undo effectué")
        else:
            print("Pas d'historique disponible")
    
    def draw_callback(self, event, x, y, flags, param):
        """Callback pour les événements de la souris"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.save_history()
            self.drawing = True
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.save_history()
            # Effacer (mettre en classe 0 = background)
            cv2.circle(self.mask, (x, y), self.brush_size, 0, -1)
            self.update_overlay()
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Dessiner avec la classe courante
                cv2.circle(self.mask, (x, y), self.brush_size, 
                          self.current_class, -1)
                self.update_overlay()
                
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Changer la taille du pinceau
            if flags > 0:
                self.brush_size = min(50, self.brush_size + 2)
            else:
                self.brush_size = max(1, self.brush_size - 2)
            print(f"Taille du pinceau: {self.brush_size}")

    
    def save_mask(self):
        image_stem = self.images[self.current_image_idx].stem

        # Sauvegarde masque multi-classes
        mask_path = self.output_dir / f"{image_stem}_mask.png"
        cv2.imwrite(str(mask_path), self.mask)

        # Sauvegarde masque binaire
        binary_mask = self.generate_binary_mask()
        binary_path = self.output_dir / f"{image_stem}_binary.png"
        cv2.imwrite(str(binary_path), binary_mask * 255)

        print(f"Masques sauvegardés: {mask_path.name}, {binary_path.name}")

    # Sauvegarde masque coloré (debug)
    color_mask = np.zeros_like(self.image)
    for class_id, color in self.colors.items():
        color_mask[self.mask == class_id] = color

    color_path = self.output_dir / f"{image_stem}_color.png"
    cv2.imwrite(str(color_path), color_mask)

    
    def run(self):
        """Lancer l'outil d'annotation"""
        if not self.images:
            print("Aucune image trouvée!")
            return
        
        cv2.namedWindow('Annotation')
        cv2.setMouseCallback('Annotation', self.draw_callback)
        
        self.load_image(0)
        show_overlay = True
        
        while True:
            # Afficher l'image
            if show_overlay:
                display = self.overlay.copy()
            else:
                display = self.image.copy()
            
            # Afficher les informations
            info_text = [
                f"Image: {self.current_image_idx + 1}/{len(self.images)}",
                f"Classe: {self.class_names[self.current_class]} ({self.current_class})",
                f"Pinceau: {self.brush_size}px",
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(display, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                y_offset += 30
            
            # Afficher la légende des classes
            legend_y = display.shape[0] - 20 * len(self.classes)
            for cls in self.classes:
                color = tuple(cls['color'])
                cv2.rectangle(display, (10, legend_y), (30, legend_y + 15), color, -1)
                cv2.putText(display, f"{cls['id']}: {cls['name']}", (35, legend_y + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(display, f"{cls['id']}: {cls['name']}", (35, legend_y + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                legend_y += 20
            
            cv2.imshow('Annotation', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Gestion des touches
            if key == ord('q'):
                break
            elif key == ord('n'):
                self.save_mask()
                self.load_image(self.current_image_idx + 1)
            elif key == ord('p'):
                self.save_mask()
                self.load_image(self.current_image_idx - 1)
            elif key == ord('s'):
                self.save_mask()
            elif key == ord('z'):
                self.undo()
            elif key == ord('c'):
                self.save_history()
                self.mask = np.zeros_like(self.mask)
                self.update_overlay()
                print("Masque effacé")
            elif key == ord('t'):
                show_overlay = not show_overlay
            elif ord('0') <= key <= ord('9'):
                class_id = key - ord('0')
                if class_id < len(self.classes):
                    self.current_class = class_id
                    print(f"Classe sélectionnée: {self.class_names[self.current_class]}")
        
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Outil d'annotation de segmentation sémantique")
    parser.add_argument('--images', type=str, required=True, help='Dossier contenant les images')
    parser.add_argument('--output', type=str, required=True, help='Dossier de sortie pour les masques')
    parser.add_argument('--config', type=str, default='label_config.json', 
                       help='Fichier de configuration des classes')
    
    args = parser.parse_args()
    
    annotator = SemanticAnnotator(args.images, args.output, args.config)
    annotator.run()
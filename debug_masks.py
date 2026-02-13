"""
Script de débogage pour diagnostiquer les problèmes de masques
Lance ce script AVANT toute chose pour comprendre ce qui ne va pas
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CARLA_CLASSES, CLASS_TO_BINARY, BINARY_COLORS


def inspect_mask(mask_path, show_plot=True):
    """
    Analyse complète d'un masque pour diagnostiquer les problèmes
    
    Args:
        mask_path: Chemin vers le masque (.npy ou .png)
        show_plot: Afficher la visualisation
    
    Returns:
        dict: Diagnostic complet
    """
    print(f"\n{'='*60}")
    print(f"INSPECTION DU MASQUE: {os.path.basename(mask_path)}")
    print(f"{'='*60}")

    # Charger le masque
    if mask_path.endswith('.npy'):
        mask = np.load(mask_path)
    else:
        mask = np.array(Image.open(mask_path))

    print(f"\n[INFO] Shape brut : {mask.shape}")
    print(f"[INFO] dtype       : {mask.dtype}")
    print(f"[INFO] min/max     : {mask.min()} / {mask.max()}")

    # Si le masque est en 3 canaux, vérifier quel canal contient les classes
    if mask.ndim == 3:
        print(f"\n⚠️  MASQUE EN {mask.ndim}D - analyse par canal :")
        for ch in range(mask.shape[2]):
            ch_vals = np.unique(mask[:, :, ch])
            print(f"   Canal {ch}: valeurs uniques = {ch_vals[:20]}...")
        
        print("\n  → Utilisation du canal 0 (à ajuster si besoin)")
        mask_2d = mask[:, :, 0]
    else:
        mask_2d = mask

    # Classes présentes dans ce masque
    unique_classes = np.unique(mask_2d)
    print(f"\n[CLASSES PRÉSENTES] {unique_classes}")

    # Vérifier si les valeurs sont dans la plage CARLA (0-22)
    max_expected = 22
    out_of_range = unique_classes[unique_classes > max_expected]

    if len(out_of_range) > 0:
        print(f"\n❌ PROBLÈME DÉTECTÉ: Valeurs hors de la plage CARLA (0-22) !")
        print(f"   Valeurs inattendues : {out_of_range}")
        print(f"   → Possible cause : mauvais canal CARLA extrait (palette CityScapes ?)")
        print(f"   → Fix : voir la fonction _process_semantic() dans collect_images.py")
    else:
        print(f"\n✓ Toutes les valeurs sont dans la plage CARLA (0-22)")

    # Distribution des classes
    print(f"\n[DISTRIBUTION DES CLASSES]")
    print(f"{'Classe':<6} {'Nom':<20} {'Pixels':>10} {'%':>8}  {'Binaire'}")
    print("-"*60)

    total_pixels = mask_2d.size
    road_pixels = 0
    road_binary_correct = True

    for cls in unique_classes:
        count = np.sum(mask_2d == cls)
        pct = count / total_pixels * 100
        name = CARLA_CLASSES.get(int(cls), f"INCONNU({cls})")
        binary = CLASS_TO_BINARY.get(int(cls), "NON MAPPÉ")
        binary_label = "Traversable" if binary == 0 else "Obstrué"

        # Identifier la route
        if int(cls) == 7:  # Road
            road_pixels = count
            flag = " ← ROUTE"
        elif int(cls) == 6:
            flag = " ← LIGNE"
        elif int(cls) == 13:
            flag = " ← CIEL"
        else:
            flag = ""

        print(f"{cls:<6} {name:<20} {count:>10} {pct:>7.2f}%  {binary_label}{flag}")

    # Vérification spécifique : est-ce que la route est présente ?
    if 7 not in unique_classes:
        print(f"\n❌ PROBLÈME CRITIQUE: Classe 7 (Road) ABSENTE du masque !")
        print(f"   → La route ne sera jamais verte dans le GT")
        print(f"   → Vérifiez l'extraction du channel dans collect_images.py")
    else:
        pct_road = road_pixels / total_pixels * 100
        print(f"\n✓ Route présente : {road_pixels} pixels ({pct_road:.2f}%)")

    return {
        'mask': mask_2d,
        'unique_classes': unique_classes,
        'has_road': 7 in unique_classes,
        'out_of_range': out_of_range
    }


def compare_rgb_vs_mask(image_path, mask_path, save_path=None):
    """
    Affiche l'image originale, le masque brut (toutes classes)
    et le masque binaire côte à côte pour diagnostiquer
    """
    # Charger l'image
    image = np.array(Image.open(image_path).convert('RGB'))

    # Charger le masque
    if mask_path.endswith('.npy'):
        mask = np.load(mask_path)
    else:
        mask = np.array(Image.open(mask_path))

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # Coloriser le masque multi-classe (23 couleurs)
    colors_23 = plt.cm.tab20(np.linspace(0, 1, 23))
    mask_colored = colors_23[np.clip(mask, 0, 22)]

    # Créer le masque binaire
    binary_mask = np.zeros_like(mask, dtype=np.uint8)
    for carla_class, binary_class in CLASS_TO_BINARY.items():
        binary_mask[mask == carla_class] = binary_class

    # Coloriser en vert/rouge
    binary_colored = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
    binary_colored[binary_mask == 0] = BINARY_COLORS[0]  # Vert = traversable
    binary_colored[binary_mask == 1] = BINARY_COLORS[1]  # Rouge = obstrué

    # Afficher
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title('Image RGB Originale', fontsize=14)
    axes[0].axis('off')

    # Légendes pour le masque multi-classe
    axes[1].imshow(mask_colored)
    axes[1].set_title('Masque CARLA (23 classes)\n[debug: chaque couleur = 1 classe]', fontsize=12)
    axes[1].axis('off')

    # Ajouter quelques labels de classes importantes sur le masque multi-classe
    for cls_id in [7, 6, 8, 10, 4, 9, 13]:  # Road, RoadLine, Sidewalk, Vehicle, Pedestrian, Vegetation, Sky
        positions = np.argwhere(mask == cls_id)
        if len(positions) > 100:
            cy, cx = positions.mean(axis=0).astype(int)
            axes[1].text(cx, cy, CARLA_CLASSES.get(cls_id, str(cls_id)),
                        fontsize=7, color='white',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

    axes[2].imshow(binary_colored)
    axes[2].set_title('Masque Binaire Généré\n(vert=traversable, rouge=obstrué)', fontsize=12)
    axes[2].axis('off')

    # Légende binaire
    green_patch = mpatches.Patch(color=np.array(BINARY_COLORS[0]) / 255, label='Traversable (route, trottoir…)')
    red_patch = mpatches.Patch(color=np.array(BINARY_COLORS[1]) / 255, label='Obstrué (véhicules, ciel, murs…)')
    axes[2].legend(handles=[green_patch, red_patch], loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.suptitle('Diagnostic de Segmentation', fontsize=16, y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Diagnostic sauvegardé: {save_path}")
    else:
        plt.show()
    plt.close()


def check_carla_channel(mask_raw_path):
    """
    Vérifie quel canal CARLA contient les vraies classes sémantiques
    Utile si on suspecte un mauvais canal dans collect_images.py
    
    Args:
        mask_raw_path: Chemin vers un masque .npy brut 4 canaux
    """
    if not mask_raw_path.endswith('.npy'):
        print("⚠️ Cette fonction nécessite un fichier .npy (4 canaux bruts CARLA)")
        return

    raw = np.load(mask_raw_path)
    print(f"\nShape brut: {raw.shape}")

    if raw.ndim == 2:
        print("Masque déjà 2D (un seul canal enregistré)")
        return

    print("\nAnalyse de chaque canal pour identifier les classes sémantiques :")
    print(f"{'Canal':<8} {'Valeurs uniques':<40} {'Plage CARLA?'}")
    print("-"*70)

    for ch in range(raw.shape[-1] if raw.ndim == 3 else raw.shape[0]):
        if raw.ndim == 3:
            channel_data = raw[:, :, ch]
        else:
            channel_data = raw[ch]

        unique_vals = np.unique(channel_data)
        in_carla_range = np.all(unique_vals <= 22)
        n_classes = len(unique_vals[unique_vals <= 22])

        print(f"Canal {ch:<4} {str(unique_vals[:10]):<40} {'✓ OUI' if in_carla_range else '❌ NON'} "
              f"({n_classes} classes CARLA)")

    print("\n→ Le bon canal est celui avec des valeurs 0-22 ET le plus de classes distinctes")


def visualize_class_overlay(image_path, mask_path, class_ids=None, save_path=None):
    """
    Superpose les classes spécifiques sur l'image pour vérifier leur position
    
    Args:
        image_path: Chemin vers l'image
        mask_path: Chemin vers le masque
        class_ids: Liste des IDs de classes à mettre en évidence (défaut: route + obstacles)
    """
    if class_ids is None:
        class_ids = {
            7: ('Road', (0, 255, 0)),
            6: ('RoadLine', (0, 200, 0)),
            8: ('Sidewalk', (150, 255, 150)),
            10: ('Vehicles', (255, 0, 0)),
            4: ('Pedestrian', (255, 100, 0)),
            9: ('Vegetation', (0, 100, 0)),
            13: ('Sky', (100, 100, 255)),
        }

    image = np.array(Image.open(image_path).convert('RGB')).copy()

    if mask_path.endswith('.npy'):
        mask = np.load(mask_path)
    else:
        mask = np.array(Image.open(mask_path))

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # Resize le masque à la taille de l'image
    if mask.shape != image.shape[:2]:
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray(mask.astype(np.uint8))
        mask_pil = mask_pil.resize((image.shape[1], image.shape[0]), PILImage.NEAREST)
        mask = np.array(mask_pil)

    # Créer l'overlay
    overlay = image.copy()
    alpha = 0.5

    legend_patches = []
    for cls_id, (name, color) in class_ids.items():
        region = mask == cls_id
        if region.any():
            overlay[region] = (np.array(color) * alpha + overlay[region] * (1 - alpha)).astype(np.uint8)
            patch = mpatches.Patch(color=np.array(color) / 255, label=f'{cls_id}: {name}')
            legend_patches.append(patch)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(image)
    axes[0].set_title('Image Originale', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title('Overlay des classes sémantiques\n(vérification du mapping)', fontsize=12)
    axes[1].axis('off')
    axes[1].legend(handles=legend_patches, loc='lower right', fontsize=9,
                   title='Classes CARLA détectées')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Overlay sauvegardé: {save_path}")
    else:
        plt.show()
    plt.close()


def run_full_diagnostic(images_dir, masks_dir, n_samples=5, output_dir='debug_output'):
    """
    Lance un diagnostic complet sur plusieurs images
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(images_dir)
                          if f.endswith(('.png', '.jpg', '.jpeg'))])[:n_samples]

    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC COMPLET - {n_samples} images")
    print(f"{'='*60}")

    road_present_count = 0
    wrong_range_count = 0

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)
        base = os.path.splitext(img_file)[0]

        # Chercher le masque
        mask_path = None
        for ext in ['.npy', '.png']:
            candidate = os.path.join(masks_dir, base + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is None:
            print(f"⚠️ Masque introuvable pour {img_file}")
            continue

        # Inspecter
        result = inspect_mask(mask_path, show_plot=False)

        if result['has_road']:
            road_present_count += 1
        if len(result['out_of_range']) > 0:
            wrong_range_count += 1

        # Générer la visualisation de comparaison
        save_path = os.path.join(output_dir, f'debug_{i:03d}_{base}.png')
        compare_rgb_vs_mask(img_path, mask_path, save_path=save_path)

        # Générer l'overlay de classes
        save_overlay = os.path.join(output_dir, f'overlay_{i:03d}_{base}.png')
        visualize_class_overlay(img_path, mask_path, save_path=save_overlay)

    # Résumé
    print(f"\n{'='*60}")
    print(f"RÉSUMÉ DU DIAGNOSTIC")
    print(f"{'='*60}")
    print(f"Images analysées     : {len(image_files)}")
    print(f"Route présente (cls 7): {road_present_count}/{len(image_files)}")
    print(f"Valeurs hors plage   : {wrong_range_count}/{len(image_files)}")

    if road_present_count == 0:
        print(f"\n❌ DIAGNOSTIC: Classe Road (7) jamais présente!")
        print(f"   Cause probable: mauvais canal extrait de la caméra CARLA")
        print(f"   → Ouvrez collect_images.py et vérifiez _process_semantic()")
        print(f"   → Essayez array[:, :, 0] ou array[:, :, 1] au lieu de array[:, :, 2]")

    elif road_present_count < len(image_files) // 2:
        print(f"\n⚠️ ATTENTION: Route absente dans {len(image_files) - road_present_count} images")
        print(f"   Vérifiez que la caméra est bien positionnée sur la route")

    else:
        print(f"\n✓ Classes correctement extraites")

    if wrong_range_count > 0:
        print(f"\n❌ Valeurs hors plage CARLA détectées dans {wrong_range_count} masques")
        print(f"   Probable confusion avec la palette CityScapes")
        print(f"   → Vérifiez _process_semantic() dans collect_images.py")

    print(f"\nVisualisations sauvegardées dans: {output_dir}/")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Diagnostic des masques de segmentation')

    parser.add_argument('--images', type=str, required=True,
                       help='Dossier des images')
    parser.add_argument('--masks', type=str, required=True,
                       help='Dossier des masques')
    parser.add_argument('--n_samples', type=int, default=5,
                       help='Nombre d\'images à analyser (défaut: 5)')
    parser.add_argument('--output', type=str, default='debug_output',
                       help='Dossier de sortie pour les visualisations')
    parser.add_argument('--single_mask', type=str, default=None,
                       help='Analyser un seul masque en détail')
    parser.add_argument('--check_channel', type=str, default=None,
                       help='Vérifier les canaux d\'un masque .npy brut 4 canaux')

    args = parser.parse_args()

    if args.single_mask:
        inspect_mask(args.single_mask)
        return

    if args.check_channel:
        check_carla_channel(args.check_channel)
        return

    run_full_diagnostic(args.images, args.masks, args.n_samples, args.output)


if __name__ == '__main__':
    main()
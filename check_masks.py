"""
Script de vérification rapide des masques CARLA
Lance ce script pour diagnostiquer si vos masques contiennent bien les bonnes classes
"""

import numpy as np
import os
import sys
from glob import glob

# Ajouter le chemin parent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import CARLA_CLASSES, CLASS_TO_BINARY
except ImportError:
    print("⚠️ Impossible d'importer config.py")
    print("Assurez-vous que config.py est dans le même dossier")
    sys.exit(1)


def check_single_mask(mask_path):
    """Vérifie un seul masque"""
    if mask_path.endswith('.npy'):
        mask = np.load(mask_path)
    else:
        from PIL import Image
        mask = np.array(Image.open(mask_path))
    
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    
    unique_values = np.unique(mask)
    
    print(f"  Shape: {mask.shape}")
    print(f"  Valeurs uniques: {unique_values}")
    print(f"  Min: {mask.min()}, Max: {mask.max()}")
    
    # Vérifier les classes importantes
    has_road = 7 in unique_values
    has_unlabeled_only = len(unique_values) == 1 and unique_values[0] == 0
    out_of_range = unique_values[unique_values > 22]
    
    print(f"  Classe 7 (Road) présente: {'✅ OUI' if has_road else '❌ NON'}")
    
    if has_unlabeled_only:
        print(f"  ⚠️ ATTENTION: Le masque ne contient QUE des 0 (Unlabeled)")
        print(f"     → Problème de collecte CARLA probable")
    
    if len(out_of_range) > 0:
        print(f"  ❌ ERREUR: Valeurs hors plage CARLA (>22): {out_of_range}")
        print(f"     → Problème d'extraction du canal sémantique")
    
    # Distribution des classes
    print(f"\n  Distribution des classes:")
    total_pixels = mask.size
    for cls in unique_values:
        count = np.sum(mask == cls)
        pct = count / total_pixels * 100
        name = CARLA_CLASSES.get(int(cls), f"INCONNU({cls})")
        binary = CLASS_TO_BINARY.get(int(cls), "?")
        binary_label = "Traversable" if binary == 0 else "Obstrué"
        print(f"    Classe {cls:2d} ({name:20s}): {count:8d} pixels ({pct:5.2f}%) → {binary_label}")
    
    return has_road, has_unlabeled_only, len(out_of_range) > 0


def check_masks_directory(masks_dir, n_samples=5):
    """Vérifie plusieurs masques d'un dossier"""
    print(f"\n{'='*70}")
    print(f"VÉRIFICATION DES MASQUES")
    print(f"Dossier: {masks_dir}")
    print(f"{'='*70}\n")
    
    # Trouver les masques
    mask_files = []
    for ext in ['*.npy', '*.png']:
        mask_files.extend(glob(os.path.join(masks_dir, ext)))
    
    mask_files = sorted(mask_files)[:n_samples]
    
    if len(mask_files) == 0:
        print("❌ Aucun masque trouvé (.npy ou .png)")
        return
    
    print(f"Nombre de masques trouvés: {len(mask_files)}\n")
    
    # Statistiques
    total_with_road = 0
    total_unlabeled_only = 0
    total_out_of_range = 0
    
    for i, mask_file in enumerate(mask_files):
        print(f"[{i+1}/{len(mask_files)}] {os.path.basename(mask_file)}")
        has_road, unlabeled_only, out_of_range = check_single_mask(mask_file)
        
        if has_road:
            total_with_road += 1
        if unlabeled_only:
            total_unlabeled_only += 1
        if out_of_range:
            total_out_of_range += 1
        
        print()
    
    # Résumé
    print(f"{'='*70}")
    print(f"RÉSUMÉ")
    print(f"{'='*70}")
    print(f"Masques analysés:           {len(mask_files)}")
    print(f"Avec classe Road (7):       {total_with_road}/{len(mask_files)}")
    print(f"Uniquement Unlabeled (0):   {total_unlabeled_only}/{len(mask_files)}")
    print(f"Valeurs hors plage (>22):   {total_out_of_range}/{len(mask_files)}")
    print()
    
    # Diagnostic
    if total_with_road == 0:
        print("❌ PROBLÈME CRITIQUE: Aucun masque ne contient la classe Road (7)")
        print("   → Vos masques CARLA sont probablement corrompus")
        print("   → Vérifiez le script de collecte (collect_images.py)")
        print("   → Probablement un mauvais canal extrait de la caméra sémantique")
        print()
        print("SOLUTION:")
        print("   1. Ouvrez collect_images.py")
        print("   2. Cherchez la fonction _process_semantic()")
        print("   3. Vérifiez que vous extrayez le bon canal (souvent [:, :, 0] ou [:, :, 2])")
        print("   4. Relancez la collecte de données")
    
    elif total_unlabeled_only > 0:
        print("⚠️ ATTENTION: Certains masques ne contiennent que des 0")
        print("   → Possible problème de caméra ou de spawn dans CARLA")
        print("   → Vérifiez que la caméra est bien positionnée sur la route")
    
    elif total_out_of_range > 0:
        print("❌ ERREUR: Valeurs hors de la plage CARLA détectées")
        print("   → Confusion probable avec la palette CityScapes")
        print("   → Vérifiez l'extraction du canal dans collect_images.py")
    
    else:
        print("✅ Tous les masques semblent corrects!")
        print("   → Classes CARLA présentes et valides")
        print("   → Vous pouvez continuer l'entraînement")
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Vérification rapide des masques')
    parser.add_argument('--masks', type=str, default='data/val/masks',
                       help='Dossier des masques à vérifier')
    parser.add_argument('--n_samples', type=int, default=5,
                       help='Nombre de masques à analyser')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.masks):
        print(f"❌ Le dossier {args.masks} n'existe pas")
        sys.exit(1)
    
    check_masks_directory(args.masks, args.n_samples)
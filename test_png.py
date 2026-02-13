#!/usr/bin/env python
"""
Test rapide : vérifier que le chargement des masques est correct
après correction du bug .convert('L')
"""

import numpy as np
from PIL import Image
import sys

def test_png_loading_methods(png_path):
    """Compare les deux méthodes de chargement"""
    
    print(f"\n{'='*60}")
    print(f"TEST DE CHARGEMENT PNG")
    print(f"Fichier: {png_path}")
    print(f"{'='*60}\n")
    
    # Méthode 1 : BUGUÉE (convert('L'))
    print("1️⃣ Méthode BUGUÉE : convert('L')")
    mask_pil = Image.open(png_path)
    print(f"   Mode PIL original: {mask_pil.mode}")
    
    if mask_pil.mode != 'L':
        mask_pil = mask_pil.convert('L')
    mask_buggy = np.array(mask_pil)
    
    print(f"   Shape après convert: {mask_buggy.shape}")
    print(f"   Valeurs uniques: {np.unique(mask_buggy)}")
    print(f"   Min/Max: {mask_buggy.min()} / {mask_buggy.max()}")
    
    has_road_buggy = 7 in np.unique(mask_buggy)
    print(f"   Classe 7 (Road) présente: {'✅' if has_road_buggy else '❌'}")
    
    # Méthode 2 : CORRECTE (canal 0)
    print("\n2️⃣ Méthode CORRECTE : extraction canal 0")
    mask_pil = Image.open(png_path)
    mask_correct = np.array(mask_pil)
    print(f"   Shape brut: {mask_correct.shape}")
    
    if len(mask_correct.shape) == 3:
        print(f"   → Extraction du canal 0")
        mask_correct = mask_correct[:, :, 0]
    
    print(f"   Shape final: {mask_correct.shape}")
    print(f"   Valeurs uniques: {np.unique(mask_correct)}")
    print(f"   Min/Max: {mask_correct.min()} / {mask_correct.max()}")
    
    has_road_correct = 7 in np.unique(mask_correct)
    print(f"   Classe 7 (Road) présente: {'✅' if has_road_correct else '❌'}")
    
    # Comparaison
    print(f"\n{'='*60}")
    print("COMPARAISON")
    print(f"{'='*60}")
    
    if has_road_buggy and has_road_correct:
        print("✅ Les deux méthodes fonctionnent (masque déjà en grayscale)")
    elif not has_road_buggy and has_road_correct:
        print("❌ Méthode buguée détecte PAS Road")
        print("✅ Méthode correcte détecte Road")
        print("\n⚠️  CONFIRMATION DU BUG : Le PNG est en RGB et convert('L') corrompt les classes")
        
        # Montrer l'effet de la corruption
        print("\nEFFET DE LA CORRUPTION (exemples) :")
        for orig_cls in [7, 6, 10]:  # Road, RoadLine, Vehicles
            if orig_cls in np.unique(mask_correct):
                # Dans le masque bugué, classe 7 devient ~2.1 → 2
                corrupted_val = int(0.299 * orig_cls)
                print(f"  Classe {orig_cls:2d} → {corrupted_val:2d} (0.299 * {orig_cls} ≈ {0.299*orig_cls:.1f})")
    elif has_road_buggy and not has_road_correct:
        print("⚠️  Cas inattendu : seule la méthode buguée détecte Road")
        print("    Vérifiez le canal contenant les classes")
    else:
        print("❌ Aucune méthode ne détecte Road - problème de masque")
    
    print(f"\n{'='*60}\n")
    
    return has_road_buggy, has_road_correct


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_png_loading.py <masque.png>")
        print("\nExemple:")
        print("  python test_png_loading.py data/val/masks/F70-27.png")
        sys.exit(1)
    
    png_path = sys.argv[1]
    
    try:
        has_road_buggy, has_road_correct = test_png_loading_methods(png_path)
        
        if not has_road_buggy and has_road_correct:
            print("✅ CONFIRMATION : Le bug .convert('L') est présent dans votre code")
            print("📝 ACTION : Remplacez dataset.py par la version corrigée")
        elif has_road_buggy and has_road_correct:
            print("✅ OK : Votre PNG est déjà en grayscale, pas de corruption")
        else:
            print("⚠️  ATTENTION : Résultats inattendus, vérifiez le masque")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
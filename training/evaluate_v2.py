"""
Évaluation v2 pour ENet.

Points clés :
- détecte automatiquement le variant ENet du checkpoint
  - ENet "current" du projet (clés down1_0 / reg1_1 / deconv)
  - ENet "classic" du checkpoint (clés bottleneck1_0 / fullconv)
- recharge les anciens checkpoints qui référencent training.config.TrainingConfig
- corrige l'évaluation single image en appliquant la même transformation
  géométrique au masque GT puis en le binarisant APRÈS transformation
- conserve l'évaluation dataset complet + sauvegarde des visualisations
"""

import os
import sys
import json
import types
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Chemins et compatibilité imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "config.py").exists() else SCRIPT_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports projet : version structurée du repo en priorité, puis fallback flat
import config as root_config

try:
    from models.enet import get_enet_model
    from data.dataset import SegmentationDataset, get_validation_augmentation
    from utils.metrics import SegmentationMetrics
    from utils.visualization import visualize_predictions, colorize_mask
except ImportError:
    from enet import get_enet_model
    from dataset import SegmentationDataset, get_validation_augmentation
    from metrics import SegmentationMetrics
    from visualization import visualize_predictions, colorize_mask

# ---------------------------------------------------------------------------
# Compatibilité anciens checkpoints picklés avec training.config.TrainingConfig
# ---------------------------------------------------------------------------
training_pkg = sys.modules.setdefault("training", types.ModuleType("training"))
training_config_module = types.ModuleType("training.config")

for _name in dir(root_config):
    if not _name.startswith("__"):
        setattr(training_config_module, _name, getattr(root_config, _name))

class TrainingConfig:
    """Stub minimal pour recharger les anciens checkpoints picklés."""
    pass

training_config_module.TrainingConfig = TrainingConfig
sys.modules["training.config"] = training_config_module
setattr(training_pkg, "config", training_config_module)


# ---------------------------------------------------------------------------
# ENet classique correspondant au checkpoint fourni
# ---------------------------------------------------------------------------
class InitialBlockClassic(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels - in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        conv = self.conv(x)
        pool = self.pool(x)
        out = torch.cat([conv, pool], dim=1)
        out = self.bn(out)
        out = self.prelu(out)
        return out


class RegularBottleneckClassic(nn.Module):
    def __init__(
        self,
        channels,
        internal_ratio=4,
        kernel_size=3,
        padding=1,
        dilation=1,
        asymmetric=False,
        dropout_prob=0.1,
    ):
        super().__init__()
        internal_channels = channels // internal_ratio

        self.conv1 = nn.Conv2d(channels, internal_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = nn.PReLU()

        if asymmetric:
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    padding=(padding, 0),
                    bias=False,
                ),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU(),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    padding=(0, padding),
                    bias=False,
                ),
            )
        else:
            self.conv2 = nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            )

        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(internal_channels, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.prelu3 = nn.PReLU()
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)

        out = out + identity
        out = self.prelu3(out)
        return out


class DownsamplingBottleneckClassic(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, dropout_prob=0.1):
        super().__init__()
        internal_channels = in_channels // internal_ratio

        self.conv1 = nn.Conv2d(
            in_channels,
            internal_channels,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(
            internal_channels,
            internal_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(
            internal_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.prelu3 = nn.PReLU()
        self.dropout = nn.Dropout2d(p=dropout_prob)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)

        skip, indices = self.pool(x)
        skip = self.skip_conv(skip)

        out = out + skip
        out = self.prelu3(out)
        return out, indices


class UpsamplingBottleneckClassic(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, dropout_prob=0.1):
        super().__init__()
        internal_channels = in_channels // internal_ratio

        self.conv1 = nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = nn.PReLU()

        self.conv2 = nn.ConvTranspose2d(
            internal_channels,
            internal_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.prelu3 = nn.PReLU()
        self.dropout = nn.Dropout2d(p=dropout_prob)

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x, indices):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)

        skip = self.skip_conv(x)
        skip = self.unpool(skip, indices)

        out = out + skip
        out = self.prelu3(out)
        return out


class ENetClassic(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.initial = InitialBlockClassic(3, 16)

        self.bottleneck1_0 = DownsamplingBottleneckClassic(16, 64, dropout_prob=0.01)
        self.bottleneck1_1 = RegularBottleneckClassic(64, dropout_prob=0.01)
        self.bottleneck1_2 = RegularBottleneckClassic(64, dropout_prob=0.01)
        self.bottleneck1_3 = RegularBottleneckClassic(64, dropout_prob=0.01)
        self.bottleneck1_4 = RegularBottleneckClassic(64, dropout_prob=0.01)

        self.bottleneck2_0 = DownsamplingBottleneckClassic(64, 128, dropout_prob=0.1)
        self.bottleneck2_1 = RegularBottleneckClassic(128, dropout_prob=0.1)
        self.bottleneck2_2 = RegularBottleneckClassic(128, padding=2, dilation=2, dropout_prob=0.1)
        self.bottleneck2_3 = RegularBottleneckClassic(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.bottleneck2_4 = RegularBottleneckClassic(128, padding=4, dilation=4, dropout_prob=0.1)
        self.bottleneck2_5 = RegularBottleneckClassic(128, dropout_prob=0.1)
        self.bottleneck2_6 = RegularBottleneckClassic(128, padding=8, dilation=8, dropout_prob=0.1)
        self.bottleneck2_7 = RegularBottleneckClassic(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.bottleneck2_8 = RegularBottleneckClassic(128, padding=16, dilation=16, dropout_prob=0.1)

        self.bottleneck3_0 = RegularBottleneckClassic(128, dropout_prob=0.1)
        self.bottleneck3_1 = RegularBottleneckClassic(128, padding=2, dilation=2, dropout_prob=0.1)
        self.bottleneck3_2 = RegularBottleneckClassic(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.bottleneck3_3 = RegularBottleneckClassic(128, padding=4, dilation=4, dropout_prob=0.1)
        self.bottleneck3_4 = RegularBottleneckClassic(128, dropout_prob=0.1)
        self.bottleneck3_5 = RegularBottleneckClassic(128, padding=8, dilation=8, dropout_prob=0.1)
        self.bottleneck3_6 = RegularBottleneckClassic(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.bottleneck3_7 = RegularBottleneckClassic(128, padding=16, dilation=16, dropout_prob=0.1)

        self.bottleneck4_0 = UpsamplingBottleneckClassic(128, 64, dropout_prob=0.1)
        self.bottleneck4_1 = RegularBottleneckClassic(64, dropout_prob=0.1)
        self.bottleneck4_2 = RegularBottleneckClassic(64, dropout_prob=0.1)

        self.bottleneck5_0 = UpsamplingBottleneckClassic(64, 16, dropout_prob=0.1)
        self.bottleneck5_1 = RegularBottleneckClassic(16, dropout_prob=0.1)

        self.fullconv = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        x = self.initial(x)

        x, indices1 = self.bottleneck1_0(x)
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x)
        x = self.bottleneck1_4(x)

        x, indices2 = self.bottleneck2_0(x)
        x = self.bottleneck2_1(x)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        x = self.bottleneck2_4(x)
        x = self.bottleneck2_5(x)
        x = self.bottleneck2_6(x)
        x = self.bottleneck2_7(x)
        x = self.bottleneck2_8(x)

        x = self.bottleneck3_0(x)
        x = self.bottleneck3_1(x)
        x = self.bottleneck3_2(x)
        x = self.bottleneck3_3(x)
        x = self.bottleneck3_4(x)
        x = self.bottleneck3_5(x)
        x = self.bottleneck3_6(x)
        x = self.bottleneck3_7(x)

        x = self.bottleneck4_0(x, indices2)
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)

        x = self.bottleneck5_0(x, indices1)
        x = self.bottleneck5_1(x)

        x = self.fullconv(x)
        return x


# ---------------------------------------------------------------------------
# Helpers de chargement / détection checkpoint
# ---------------------------------------------------------------------------
def load_checkpoint_compat(checkpoint_path, device):
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def get_state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    raise KeyError("Impossible de trouver 'model_state_dict' ou 'state_dict' dans le checkpoint.")


def detect_enet_variant(state_dict):
    keys = list(state_dict.keys())
    if any(k.startswith("bottleneck1_0") or k.startswith("fullconv") for k in keys):
        return "classic"
    if any(k.startswith("down1_0") or k.startswith("deconv") for k in keys):
        return "current"
    raise RuntimeError("Variant ENet inconnu: clés du checkpoint non reconnues.")


def build_enet_from_checkpoint(state_dict, forced_variant="auto"):
    variant = forced_variant
    if variant == "auto":
        variant = detect_enet_variant(state_dict)

    if variant == "classic":
        model = ENetClassic(num_classes=2)
    elif variant == "current":
        model = get_enet_model(num_classes=2)
    else:
        raise ValueError(f"Variant ENet invalide: {variant}")

    model.load_state_dict(state_dict)
    return model, variant


# ---------------------------------------------------------------------------
# Helpers masque / visualisation / métriques
# ---------------------------------------------------------------------------
def convert_carla_mask_to_binary(mask_np):
    mask_np = mask_np.astype(np.int32)
    binary_mask = np.ones(mask_np.shape, dtype=np.uint8)
    for carla_class, binary_class in root_config.CLASS_TO_BINARY.items():
        binary_mask[mask_np == carla_class] = binary_class
    return binary_mask


def load_raw_mask(mask_path):
    if mask_path.endswith(".npy"):
        mask = np.load(mask_path)
    else:
        mask = np.array(Image.open(mask_path))

    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def save_metrics_json(summary, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_ready = {}
    for key, value in summary.items():
        if isinstance(value, np.ndarray):
            json_ready[key] = value.tolist()
        else:
            json_ready[key] = value
    out_path = os.path.join(output_dir, "metrics_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=2, ensure_ascii=False)
    return out_path


# ---------------------------------------------------------------------------
# Évaluation dataset complet
# ---------------------------------------------------------------------------
def evaluate_model(model, dataloader, device, save_predictions=False, output_dir=None):
    model.eval()
    metrics = SegmentationMetrics(num_classes=2)

    cached_images = []
    cached_masks = []
    cached_predictions = []

    print("\nÉvaluation en cours...")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            metrics.update(predictions, masks)

            if save_predictions and batch_idx < 10:
                cached_images.extend(images.cpu())
                cached_masks.extend(masks.cpu())
                cached_predictions.extend(predictions.cpu())

    print("\n" + "=" * 60)
    print("RÉSULTATS DE L'ÉVALUATION")
    print("=" * 60)
    metrics.print_summary(class_names=list(root_config.BINARY_CLASSES.values()))

    summary = metrics.get_summary()

    if output_dir:
        metrics_path = save_metrics_json(summary, output_dir)
        print(f"Résumé JSON sauvegardé: {metrics_path}")

    if save_predictions and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for i in range(0, min(len(cached_images), 40), 4):
            end_idx = min(i + 4, len(cached_images))
            save_path = os.path.join(output_dir, f"predictions_{i // 4:03d}.png")
            visualize_predictions(
                torch.stack(cached_images[i:end_idx]),
                torch.stack(cached_masks[i:end_idx]),
                torch.stack(cached_predictions[i:end_idx]),
                num_samples=end_idx - i,
                save_path=save_path,
            )
        print(f"Visualisations sauvegardées dans {output_dir}")

    return summary


# ---------------------------------------------------------------------------
# Évaluation image unique
# ---------------------------------------------------------------------------
def evaluate_single_image(model, image_path, mask_path, device, save_path=None):
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    transform = get_validation_augmentation(
        root_config.TRAINING_IMAGE_SIZE,
        root_config.PRESERVE_ASPECT_RATIO,
    )

    dummy_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    transformed = transform(image=image_np, mask=dummy_mask)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).squeeze(0).cpu()

    gt_mask_tensor = None
    if mask_path and os.path.exists(mask_path):
        raw_mask = load_raw_mask(mask_path)
        transformed_mask = transform(image=image_np, mask=raw_mask)["mask"]

        if isinstance(transformed_mask, torch.Tensor):
            transformed_mask = transformed_mask.cpu().numpy()

        binary_mask = convert_carla_mask_to_binary(transformed_mask)
        gt_mask_tensor = torch.from_numpy(binary_mask).long()

        metrics = SegmentationMetrics(num_classes=2)
        metrics.update(prediction.unsqueeze(0), gt_mask_tensor.unsqueeze(0))

        print("\n" + "=" * 60)
        print("MÉTRIQUES POUR L'IMAGE")
        print("=" * 60)
        metrics.print_summary(class_names=list(root_config.BINARY_CLASSES.values()))

    os.makedirs(os.path.dirname(save_path) if save_path else ".", exist_ok=True)

    if gt_mask_tensor is not None:
        visualize_predictions(
            input_tensor.cpu(),
            gt_mask_tensor.unsqueeze(0),
            prediction.unsqueeze(0),
            num_samples=1,
            save_path=save_path,
        )
    else:
        import matplotlib.pyplot as plt

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_denorm = input_tensor.cpu() * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        img_denorm = img_denorm.squeeze(0).permute(1, 2, 0).numpy()

        pred_colored = colorize_mask(prediction.numpy())

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_denorm)
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(pred_colored)
        axes[1].set_title("Prédiction")
        axes[1].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Visualisation sauvegardée: {save_path}")
        else:
            plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Évaluation v2 pour ENet")
    parser.add_argument("--checkpoint", type=str, required=True, help="Chemin vers le checkpoint")
    parser.add_argument(
        "--enet_variant",
        type=str,
        default="auto",
        choices=["auto", "classic", "current"],
        help="Variant ENet à utiliser",
    )
    parser.add_argument("--images", type=str, default="data/test/images", help="Dossier des images de test")
    parser.add_argument("--masks", type=str, default="data/test/masks", help="Dossier des masques de test")
    parser.add_argument("--batch_size", type=int, default=8, help="Taille du batch")
    parser.add_argument("--num_workers", type=int, default=0, help="Nombre de workers DataLoader")
    parser.add_argument("--save_predictions", action="store_true", help="Sauvegarder les visualisations")
    parser.add_argument("--output", type=str, default="evaluation_results_v2", help="Dossier de sortie")
    parser.add_argument("--single_image", type=str, default=None, help="Évaluer une image unique")
    parser.add_argument("--single_mask", type=str, default=None, help="Masque GT d'une image unique")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nChargement du checkpoint depuis {args.checkpoint}...")
    checkpoint = load_checkpoint_compat(args.checkpoint, device)
    state_dict = get_state_dict_from_checkpoint(checkpoint)

    model, resolved_variant = build_enet_from_checkpoint(state_dict, args.enet_variant)
    model = model.to(device)
    model.eval()

    print(f"Variant ENet chargé: {resolved_variant}")
    print("Modèle chargé avec succès !")

    if args.single_image:
        os.makedirs(args.output, exist_ok=True)
        save_path = os.path.join(args.output, "single_prediction.png")
        evaluate_single_image(
            model=model,
            image_path=args.single_image,
            mask_path=args.single_mask,
            device=device,
            save_path=save_path,
        )
        return

    dataset = SegmentationDataset(
        images_dir=args.images,
        masks_dir=args.masks,
        transform=get_validation_augmentation(
            root_config.TRAINING_IMAGE_SIZE,
            root_config.PRESERVE_ASPECT_RATIO,
        ),
        binary_output=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Dataset de test: {len(dataset)} images")
    evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        save_predictions=args.save_predictions,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()

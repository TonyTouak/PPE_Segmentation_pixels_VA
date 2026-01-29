"""
U-Net Architecture pour Segmentation Sémantique
Architecture avec skip connections pour préserver les détails

Référence: U-Net: Convolutional Networks for Biomedical Image Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling: MaxPool -> DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling: ConvTranspose2d -> Concatenate -> DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                        kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Ajuster la taille si nécessaire
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Convolution finale pour la sortie"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net pour segmentation sémantique
    Deux modes:
    - num_classes=23: Segmentation multi-classe CARLA
    - num_classes=2: Segmentation binaire directe
    """
    def __init__(self, num_classes=2, in_channels=3, bilinear=False):
        super(UNet, self).__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output
        self.outc = OutConv(64, num_classes)
    
    def forward(self, x):
        # Encoder avec skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder avec skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits


class UNetSmall(nn.Module):
    """
    Version légère de U-Net pour un entraînement plus rapide
    """
    def __init__(self, num_classes=2, in_channels=3):
        super(UNetSmall, self).__init__()
        
        self.num_classes = num_classes
        
        # Encoder
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        
        # Decoder
        self.up1 = Up(256, 128, bilinear=False)
        self.up2 = Up(128, 64, bilinear=False)
        self.up3 = Up(64, 32, bilinear=False)
        
        # Output
        self.outc = OutConv(32, num_classes)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits


def get_unet_model(num_classes=2, model_type='standard', bilinear=False):
    """
    Crée un modèle U-Net
    
    Args:
        num_classes: Nombre de classes (2 pour binaire, 23 pour multi-classe)
        model_type: 'standard' ou 'small'
        bilinear: Utiliser l'upsampling bilinéaire au lieu de ConvTranspose
    
    Returns:
        model: Modèle U-Net
    """
    if model_type == 'small':
        model = UNetSmall(num_classes=num_classes)
    else:
        model = UNet(num_classes=num_classes, bilinear=bilinear)
    
    return model


if __name__ == "__main__":
    # Test des modèles
    print("=== U-Net Standard ===")
    model = get_unet_model(num_classes=2, model_type='standard')
    print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    print("\n=== U-Net Small ===")
    model_small = get_unet_model(num_classes=2, model_type='small')
    print(f"Nombre de paramètres: {sum(p.numel() for p in model_small.parameters()):,}")
    
    y_small = model_small(x)
    print(f"Output shape: {y_small.shape}")
"""
ENet Architecture pour Segmentation Sémantique
Architecture légère et rapide pour temps réel

Référence: ENet: A Deep Neural Network Architecture for Real-time Semantic Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InitialBlock(nn.Module):
    """Bloc initial - downsampling agressif"""
    def __init__(self, in_channels=3, out_channels=16):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, 
                             kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Branche convolution
        conv_out = self.conv(x)
        # Branche pooling
        pool_out = self.pool(x)
        # Concaténation
        out = torch.cat([conv_out, pool_out], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out


class BottleneckDownsampling(nn.Module):
    """Bottleneck avec downsampling"""
    def __init__(self, in_channels, out_channels, dropout_prob=0.1):
        super(BottleneckDownsampling, self).__init__()
        
        internal_channels = in_channels // 4
        
        # Main branch
        self.conv1 = nn.Conv2d(in_channels, internal_channels, 
                              kernel_size=2, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        
        self.conv2 = nn.Conv2d(internal_channels, internal_channels, 
                              kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(internal_channels)
        
        self.conv3 = nn.Conv2d(internal_channels, out_channels, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        
        # Skip connection avec pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv_skip = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, bias=False)
        self.bn_skip = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        # Main branch
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)
        
        # Skip connection
        skip, indices = self.pool(x)
        skip = self.conv_skip(skip)
        skip = self.bn_skip(skip)
        
        # Addition
        out = out + skip
        out = self.relu(out)
        
        return out, indices


class BottleneckRegular(nn.Module):
    """Bottleneck régulier (sans downsampling)"""
    def __init__(self, channels, dropout_prob=0.1, dilation=1):
        super(BottleneckRegular, self).__init__()
        
        internal_channels = channels // 4
        
        self.conv1 = nn.Conv2d(channels, internal_channels, 
                              kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        
        self.conv2 = nn.Conv2d(internal_channels, internal_channels, 
                              kernel_size=3, padding=dilation, 
                              dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(internal_channels)
        
        self.conv3 = nn.Conv2d(internal_channels, channels, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_prob)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out


class BottleneckUpsampling(nn.Module):
    """Bottleneck avec upsampling"""
    def __init__(self, in_channels, out_channels, dropout_prob=0.1):
        super(BottleneckUpsampling, self).__init__()
        
        internal_channels = in_channels // 4
        
        # Main branch
        self.conv1 = nn.Conv2d(in_channels, internal_channels, 
                              kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        
        self.deconv = nn.ConvTranspose2d(internal_channels, internal_channels,
                                        kernel_size=3, stride=2, 
                                        padding=1, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(internal_channels)
        
        self.conv3 = nn.Conv2d(internal_channels, out_channels, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        
        # Skip connection
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv_skip = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, bias=False)
        self.bn_skip = nn.BatchNorm2d(out_channels)
    
    def forward(self, x, indices):
        # Main branch
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.deconv(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)
        
        # Skip connection
        skip = self.unpool(x, indices)
        skip = self.conv_skip(skip)
        skip = self.bn_skip(skip)
        
        # Addition
        out = out + skip
        out = self.relu(out)
        
        return out


class ENet(nn.Module):
    """
    ENet pour segmentation sémantique
    Deux modes:
    - num_classes=23: Segmentation multi-classe CARLA
    - num_classes=2: Segmentation binaire directe
    """
    def __init__(self, num_classes=2, encoder_only=False):
        super(ENet, self).__init__()
        
        self.num_classes = num_classes
        
        # ====== ENCODER ======
        # Stage 0
        self.initial = InitialBlock(3, 16)
        
        # Stage 1
        self.down1_0, self.indices1 = None, None
        self.down1_0 = BottleneckDownsampling(16, 64, dropout_prob=0.01)
        self.reg1_1 = BottleneckRegular(64, dropout_prob=0.01)
        self.reg1_2 = BottleneckRegular(64, dropout_prob=0.01)
        self.reg1_3 = BottleneckRegular(64, dropout_prob=0.01)
        self.reg1_4 = BottleneckRegular(64, dropout_prob=0.01)
        
        # Stage 2
        self.down2_0 = BottleneckDownsampling(64, 128, dropout_prob=0.1)
        self.reg2_1 = BottleneckRegular(128, dropout_prob=0.1)
        self.reg2_2 = BottleneckRegular(128, dropout_prob=0.1, dilation=2)
        self.reg2_3 = BottleneckRegular(128, dropout_prob=0.1)
        self.reg2_4 = BottleneckRegular(128, dropout_prob=0.1, dilation=4)
        self.reg2_5 = BottleneckRegular(128, dropout_prob=0.1)
        self.reg2_6 = BottleneckRegular(128, dropout_prob=0.1, dilation=8)
        self.reg2_7 = BottleneckRegular(128, dropout_prob=0.1)
        self.reg2_8 = BottleneckRegular(128, dropout_prob=0.1, dilation=16)
        
        # Stage 3 (repeat of stage 2)
        self.reg3_0 = BottleneckRegular(128, dropout_prob=0.1)
        self.reg3_1 = BottleneckRegular(128, dropout_prob=0.1, dilation=2)
        self.reg3_2 = BottleneckRegular(128, dropout_prob=0.1)
        self.reg3_3 = BottleneckRegular(128, dropout_prob=0.1, dilation=4)
        self.reg3_4 = BottleneckRegular(128, dropout_prob=0.1)
        self.reg3_5 = BottleneckRegular(128, dropout_prob=0.1, dilation=8)
        self.reg3_6 = BottleneckRegular(128, dropout_prob=0.1)
        self.reg3_7 = BottleneckRegular(128, dropout_prob=0.1, dilation=16)
        
        # ====== DECODER ======
        if not encoder_only:
            # Stage 4
            self.up4_0 = BottleneckUpsampling(128, 64, dropout_prob=0.1)
            self.reg4_1 = BottleneckRegular(64, dropout_prob=0.1)
            self.reg4_2 = BottleneckRegular(64, dropout_prob=0.1)
            
            # Stage 5
            self.up5_0 = BottleneckUpsampling(64, 16, dropout_prob=0.1)
            self.reg5_1 = BottleneckRegular(16, dropout_prob=0.1)
            
            # Final upsampling
            self.deconv = nn.ConvTranspose2d(16, num_classes, 
                                            kernel_size=2, stride=2)
    
    def forward(self, x):
        # Encoder
        x = self.initial(x)
        
        # Stage 1
        x, indices1 = self.down1_0(x)
        x = self.reg1_1(x)
        x = self.reg1_2(x)
        x = self.reg1_3(x)
        x = self.reg1_4(x)
        
        # Stage 2
        x, indices2 = self.down2_0(x)
        x = self.reg2_1(x)
        x = self.reg2_2(x)
        x = self.reg2_3(x)
        x = self.reg2_4(x)
        x = self.reg2_5(x)
        x = self.reg2_6(x)
        x = self.reg2_7(x)
        x = self.reg2_8(x)
        
        # Stage 3
        x = self.reg3_0(x)
        x = self.reg3_1(x)
        x = self.reg3_2(x)
        x = self.reg3_3(x)
        x = self.reg3_4(x)
        x = self.reg3_5(x)
        x = self.reg3_6(x)
        x = self.reg3_7(x)
        
        # Decoder
        x = self.up4_0(x, indices2)
        x = self.reg4_1(x)
        x = self.reg4_2(x)
        
        x = self.up5_0(x, indices1)
        x = self.reg5_1(x)
        
        x = self.deconv(x)
        
        return x


def get_enet_model(num_classes=2, pretrained=False):
    """
    Crée un modèle ENet
    
    Args:
        num_classes: Nombre de classes (2 pour binaire, 23 pour multi-classe)
        pretrained: Charger des poids pré-entraînés (non implémenté)
    
    Returns:
        model: Modèle ENet
    """
    model = ENet(num_classes=num_classes)
    
    if pretrained:
        print("Attention: Poids pré-entraînés non disponibles pour ENet")
    
    return model


if __name__ == "__main__":
    # Test du modèle
    model = get_enet_model(num_classes=2)
    print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
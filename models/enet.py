"""
Implémentation de ENet pour la segmentation sémantique
Basé sur "ENet: A Deep Neural Network Architecture for Real-time Semantic Segmentation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InitialBlock(nn.Module):
    """Block initial d'ENet"""
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels - in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
        
    def forward(self, x):
        # Convolution branch
        conv_branch = self.conv(x)
        
        # Pooling branch
        pool_branch = self.pool(x)
        
        # Concatenate
        out = torch.cat([conv_branch, pool_branch], dim=1)
        out = self.bn(out)
        out = self.prelu(out)
        
        return out


class BottleneckBlock(nn.Module):
    """Bottleneck block d'ENet"""
    def __init__(
        self,
        in_channels,
        out_channels,
        internal_ratio=4,
        kernel_size=3,
        padding=0,
        dilation=1,
        asymmetric=False,
        downsample=False,
        upsample=False,
        dropout_prob=0.1
    ):
        super().__init__()
        
        internal_channels = in_channels // internal_ratio
        
        self.downsample = downsample
        self.upsample = upsample
        
        # Main branch
        # 1x1 projection
        if downsample:
            self.conv1 = nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=False
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=1,
                bias=False
            )
        
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = nn.PReLU()
        
        # Main convolution
        if asymmetric:
            # Convolution asymétrique
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=False
                ),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU(),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    padding=(0, padding),
                    dilation=dilation,
                    bias=False
                )
            )
        elif upsample:
            self.conv2 = nn.ConvTranspose2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False
            )
        else:
            self.conv2 = nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False
            )
        
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.prelu2 = nn.PReLU()
        
        # 1x1 expansion
        self.conv3 = nn.Conv2d(
            internal_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.prelu3 = nn.PReLU()
        self.dropout = nn.Dropout2d(p=dropout_prob)
        
        # Skip connection
        if downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        elif upsample:
            self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip_conv = None
    
    def forward(self, x, max_indices=None):
        identity = x
        
        # Main branch
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)
        
        # Skip connection
        if self.downsample:
            identity, max_indices = self.pool(identity)
        elif self.upsample:
            if max_indices is None:
                # Si pas d'indices disponibles, utiliser interpolation
                identity = F.interpolate(identity, scale_factor=2, mode='nearest')
            else:
                identity = self.unpool(identity, max_indices)
        
        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
        
        # Ajuster les dimensions si nécessaire
        if out.shape[2:] != identity.shape[2:]:
            diff_h = identity.shape[2] - out.shape[2]
            diff_w = identity.shape[3] - out.shape[3]
            out = F.pad(out, [diff_w // 2, diff_w - diff_w // 2,
                             diff_h // 2, diff_h - diff_h // 2])
        
        # Addition
        out = out + identity
        out = self.prelu3(out)
        
        if self.downsample:
            return out, max_indices
        else:
            return out


class ENet(nn.Module):
    """Architecture ENet complète"""
    def __init__(self, num_classes, encoder_only=False):
        super().__init__()
        
        self.encoder_only = encoder_only
        
        # Initial block
        self.initial = InitialBlock(3, 16)
        
        # Stage 1 - Encoder
        self.bottleneck1_0 = BottleneckBlock(16, 64, downsample=True, dropout_prob=0.01)
        self.bottleneck1_1 = BottleneckBlock(64, 64, dropout_prob=0.01)
        self.bottleneck1_2 = BottleneckBlock(64, 64, dropout_prob=0.01)
        self.bottleneck1_3 = BottleneckBlock(64, 64, dropout_prob=0.01)
        self.bottleneck1_4 = BottleneckBlock(64, 64, dropout_prob=0.01)
        
        # Stage 2 - Encoder
        self.bottleneck2_0 = BottleneckBlock(64, 128, downsample=True, dropout_prob=0.1)
        self.bottleneck2_1 = BottleneckBlock(128, 128, dropout_prob=0.1)
        self.bottleneck2_2 = BottleneckBlock(128, 128, dilation=2, padding=2, dropout_prob=0.1)
        self.bottleneck2_3 = BottleneckBlock(128, 128, asymmetric=True, kernel_size=5, padding=2, dropout_prob=0.1)
        self.bottleneck2_4 = BottleneckBlock(128, 128, dilation=4, padding=4, dropout_prob=0.1)
        self.bottleneck2_5 = BottleneckBlock(128, 128, dropout_prob=0.1)
        self.bottleneck2_6 = BottleneckBlock(128, 128, dilation=8, padding=8, dropout_prob=0.1)
        self.bottleneck2_7 = BottleneckBlock(128, 128, asymmetric=True, kernel_size=5, padding=2, dropout_prob=0.1)
        self.bottleneck2_8 = BottleneckBlock(128, 128, dilation=16, padding=16, dropout_prob=0.1)
        
        # Stage 3 - Encoder (répétition de stage 2)
        self.bottleneck3_0 = BottleneckBlock(128, 128, dropout_prob=0.1)
        self.bottleneck3_1 = BottleneckBlock(128, 128, dilation=2, padding=2, dropout_prob=0.1)
        self.bottleneck3_2 = BottleneckBlock(128, 128, asymmetric=True, kernel_size=5, padding=2, dropout_prob=0.1)
        self.bottleneck3_3 = BottleneckBlock(128, 128, dilation=4, padding=4, dropout_prob=0.1)
        self.bottleneck3_4 = BottleneckBlock(128, 128, dropout_prob=0.1)
        self.bottleneck3_5 = BottleneckBlock(128, 128, dilation=8, padding=8, dropout_prob=0.1)
        self.bottleneck3_6 = BottleneckBlock(128, 128, asymmetric=True, kernel_size=5, padding=2, dropout_prob=0.1)
        self.bottleneck3_7 = BottleneckBlock(128, 128, dilation=16, padding=16, dropout_prob=0.1)
        
        if not encoder_only:
            # Stage 4 - Decoder
            self.bottleneck4_0 = BottleneckBlock(128, 64, upsample=True, dropout_prob=0.1)
            self.bottleneck4_1 = BottleneckBlock(64, 64, dropout_prob=0.1)
            self.bottleneck4_2 = BottleneckBlock(64, 64, dropout_prob=0.1)
            
            # Stage 5 - Decoder
            self.bottleneck5_0 = BottleneckBlock(64, 16, upsample=True, dropout_prob=0.1)
            self.bottleneck5_1 = BottleneckBlock(16, 16, dropout_prob=0.1)
            
            # Fullconv
            self.fullconv = nn.ConvTranspose2d(
                16,
                num_classes,
                kernel_size=2,
                stride=2,
                bias=False
            )
    
    def forward(self, x):
        # Initial
        x = self.initial(x)
        
        # Stage 1
        x, max_indices1 = self.bottleneck1_0(x)
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x)
        x = self.bottleneck1_4(x)
        
        # Stage 2
        x, max_indices2 = self.bottleneck2_0(x)
        x = self.bottleneck2_1(x)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        x = self.bottleneck2_4(x)
        x = self.bottleneck2_5(x)
        x = self.bottleneck2_6(x)
        x = self.bottleneck2_7(x)
        x = self.bottleneck2_8(x)
        
        # Stage 3
        x = self.bottleneck3_0(x)
        x = self.bottleneck3_1(x)
        x = self.bottleneck3_2(x)
        x = self.bottleneck3_3(x)
        x = self.bottleneck3_4(x)
        x = self.bottleneck3_5(x)
        x = self.bottleneck3_6(x)
        x = self.bottleneck3_7(x)
        
        if self.encoder_only:
            return x
        
        # Stage 4
        x = self.bottleneck4_0(x, max_indices2)
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)
        
        # Stage 5
        x = self.bottleneck5_0(x, max_indices1)
        x = self.bottleneck5_1(x)
        
        # Fullconv
        x = self.fullconv(x)
        
        return x


if __name__ == "__main__":
    # Test du modèle
    model = ENet(num_classes=2)
    x = torch.randn(1, 3, 512, 512)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Compter les paramètres
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
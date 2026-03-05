import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    # Weakened U-Net: Reduced init_features from 32 to 8 to limit capacity
    def __init__(self, in_channels=2, out_channels=1, init_features=8):
        super(UNet3D, self).__init__()
        
        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = UNet3D._block(features * 4, features * 8, name="bottleneck")
        
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
        
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
        
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")
        
        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(num_features=features),
            nn.ReLU(inplace=True),
        )

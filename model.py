import torch
import torch.nn as nn
from args import *

class Discriminator(nn.Module):
    def __init__(self, channel_img, features_dimension):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(nn.Conv2d(channels_img, features_d, kernel_size = 5, stride =2, padding=1
                ), 
                nn.LeakyReLU(0.2),
                self._block(features_dimension, features_dimension*2, 4,2,1)
                self._block(features_dimension*2, features_dimension*4, 4,2,1)
                self._block(features_dimension*4, features_dimension*8, 4,2,1)
                nn.Conv2d(features_dimension*8, 1 kernel_size = 4, stride=2, padding=0), #Discrimating fake or real
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2))
            
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(z_dim, features_g*16,4,1,0),
            self._block(features_g*16, features_g*8, 4, 2, 1),
            self._block(features_g*8, features_g*4,4,2,1),
            self._block(features_g*8, features_g*4,4,2,1),
            nn.ConvTranspose2d(
                feature_g*2, channels_img, kernel_size = 4, stride=2, padding=1),
            nn.Tan(),
            )
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU),
    def forward(self, x):
        return self.net(x)



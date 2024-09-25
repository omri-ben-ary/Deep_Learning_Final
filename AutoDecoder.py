import torch
import torch.nn as nn

class AutoDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, layers=None):
        super().__init__()
        if layers is None:
            self.decoder = nn.Sequential(
                nn.Linear(64, 256), # Default latent dim is 64
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 28 * 28)
            )
        else:
            self.decoder = nn.Sequential(layers)
    
    def forward(self, z):
        x = self.decoder(z)
        x = x.view(-1, 1, 28, 28)
        return x
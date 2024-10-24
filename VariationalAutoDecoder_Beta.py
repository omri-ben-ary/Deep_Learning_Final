import torch
import torch.nn as nn

class VariationalAutoDecoder(nn.Module):
    def __init__(self,latent_dim=128, device = "cpu"):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 7 * 7 * 256),
            nn.ReLU(),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1)
        )

    def reparameterize(self, latent_tensor):
        mu = latent_tensor[:, 0, :]
        sigma = latent_tensor[:, 1, :]
        eps = torch.randn_like(sigma).to(self.device)
        return torch.sigmoid(mu + eps * sigma)

    def forward(self, latent_tensor):
        z = self.reparameterize(latent_tensor)
        rec_image = self.decoder(z)
        rec_image = rec_image.view(-1, 28, 28)
        return rec_image




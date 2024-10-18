import torch
import torch.nn as nn

class VariationalAutoDecoder(nn.Module):
    def __init__(self,mu_layers=None, var_layers=None, device = "cpu"):
        super().__init__()
        self.device = device
        self.mu = None
        self.log_var = None
        
        if mu_layers is None:
            self.mu_net = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        else:
             self.mu_net = mu_layers

        if var_layers is None:
            self.log_var_net = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        else:
            self.log_var_net = var_layers

        self.decoder = nn.Sequential(
            nn.Linear(128, 7 * 7 * 256),
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

    def reparameterize(self):
        eps = torch.randn((1, 128)).to(self.device)
        std = torch.exp(0.5 * self.log_var)
        return self.mu + eps * std

    def forward(self, latent_vec):
        self.mu = self.mu_net(latent_vec)
        self.log_var = self.log_var_net(latent_vec)
        z = self.reparameterize()
        rec_image = self.decoder(z)
        rec_image = rec_image.view(-1, 28, 28)
        return rec_image




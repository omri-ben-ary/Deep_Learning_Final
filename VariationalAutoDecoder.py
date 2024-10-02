import torch
import torch.nn as nn

class VariationalAutoDecoder(nn.Module):
    def __init__(self, mu_layers=None, var_layers=None):
        super().__init__()

        if mu_layers is None:
            self.mu_net = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
            )
        else:
             self.mu_net = mu_layers

        if var_layers is None:
            self.log_var_net = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
            )
        else:
            self.log_var_net = var_layers
        
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 784)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std  # Reparameterization trick
    
    def forward(self, latent_vec):
        mu = self.mu_net(latent_vec)
        log_var = self.log_var_net(latent_vec) 
        z = self.reparameterize(mu, log_var)
        
        rec_image = self.decoder(z)
        rec_image = rec_image.view(-1, 28, 28)
        return rec_image, mu, log_var




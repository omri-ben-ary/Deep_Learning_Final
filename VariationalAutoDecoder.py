import torch
import torch.nn as nn

class VariationalAutoDecoder(nn.Module):
    def __init__(self,mu_layers=None, var_layers=None):
        super().__init__()
        self.mu = None
        self.log_var = None
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
            nn.ConvTranspose2d(1, 32, kernel_size=4, stride=2, padding=1),  # Output: (64, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # Output: (64, 16, 64, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),     # Output: (64, 8, 128, 128)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),      # Output: (64, 1, 256, 256)
            nn.ReLU(),
            
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)  # Final layer to reduce to (64, 1, 28, 28)
        )



    def reparameterize(self, latent_vec):
        std = torch.exp(0.5 * self.log_var)
        # print(f"mu is {self.mu}  ")
        # eps = torch.randn_like(std)
        return self.mu + latent_vec * std 
       
    
    def forward(self, latent_vec):
        self.mu = self.mu_net(latent_vec)
        self.log_var = self.log_var_net(latent_vec) 
        z = self.reparameterize(latent_vec).view(-1,1,16,16)
        print(z.shape)
        rec_image = self.decoder(z)
        rec_image = rec_image.view(-1,1, 28, 28)
        return rec_image




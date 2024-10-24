import torch
import torch.nn as nn
import torch.optim as optim


class VAD_Trainer:
    def __init__(self, var_decoder, dataloader, latent_dim=128, beta=1.0, device='cpu', lr=1e-3):
        """
        Initialize the Trainer class.
        
        :param var_decoder: The decoder model (PyTorch nn.Module)
        :param latent_dim: The size of the latent space
        :param dataloader: The DataLoader for Fashion MNIST dataset
        :param device: 'cpu' or 'cuda' (for GPU support)
        :param lr: Learning rate for the optimizer
        """
        self.var_decoder = var_decoder.to(device)
        self.latent_dim = latent_dim
        self.beta = beta
        self.dataloader = dataloader
        self.device = device
        mu = torch.randn(len(dataloader.dataset), latent_dim, device=device, requires_grad=True)
        sigma = torch.randn(len(dataloader.dataset), latent_dim, device=device, requires_grad=True)
        self.latents = torch.nn.parameter.Parameter(torch.stack([mu, sigma], dim=1)).to(device)
        
        self.optimizer = torch.optim.Adam([
            {'params': self.var_decoder.parameters(), 'lr': lr},
            {'params': self.latents, 'lr': lr}
        ])
        
        self.loss = self.elbo_loss
        
    def train_epoch(self):
        self.var_decoder.train()
        running_loss = 0.0
        
        for batch_idx, (i, x) in enumerate(self.dataloader):
            images = x.to(self.device).float()
            batch_size = images.size(0)
            z = self.latents[i,:,:]
            reconstructed_images = self.var_decoder(z)
            
            loss = self.loss(images, reconstructed_images, z[:,0,:], z[:,1,:])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        # Average loss over the epoch
        epoch_loss = running_loss / len(self.dataloader)
        return epoch_loss
    
    def train(self, num_epochs=1000, early_stopping=None):
        """
        Train the AutoDecoder for multiple epochs.
        
        :param num_epochs: Number of epochs to train the model
        """
        losses=list()
        best_loss = None
        for epoch in range(num_epochs):
            epoch_loss = self.train_epoch()
            losses.append(epoch_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            
        return losses

    def elbo_loss(self, x, x_rec, mu, sigma):
        batch_size = x.size(0)
        rec_loss = nn.functional.mse_loss(x_rec, x, reduction='sum') / batch_size 
        log_var = torch.log(sigma.pow(2))
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - sigma.pow(2), dim=1)
        return  rec_loss +  self.beta * kl_loss.mean()

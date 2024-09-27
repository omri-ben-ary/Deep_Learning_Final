import torch
import torch.nn as nn
import torch.optim as optim

def reconstruction_loss(x, x_rec):
    """
    :param x: the original images
    :param x_rec: the reconstructed images
    :return: the reconstruction loss
    """
    return torch.norm(x - x_rec) / torch.prod(torch.tensor(x.shape))

class AD_Trainer:
    def __init__(self, decoder, dataloader, latent_dim=64, device='cpu', lr=1e-3):
        """
        Initialize the Trainer class.
        
        :param decoder: The decoder model (PyTorch nn.Module)
        :param latent_dim: The size of the latent space
        :param dataloader: The DataLoader for Fashion MNIST dataset
        :param device: 'cpu' or 'cuda' (for GPU support)
        :param lr: Learning rate for the optimizer
        """
        self.decoder = decoder.to(device)
        self.latent_dim = latent_dim
        self.dataloader = dataloader
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        
        # Loss function
        self.loss = reconstruction_loss
        
    def train_epoch(self):
        self.decoder.train()
        running_loss = 0.0
        
        for _, x in self.dataloader:
            images = x.to(self.device)
            batch_size = images.size(0)
            
            # Randomly initialize latent codes
            z = torch.randn((batch_size, self.latent_dim)).to(self.device)
            
            reconstructed_images = self.decoder(z)
            loss = self.loss(images, reconstructed_images)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        
        # Average loss over the epoch
        epoch_loss = running_loss / len(self.dataloader)
        return epoch_loss
    
    def train(self, num_epochs):
        """
        Train the AutoDecoder for multiple epochs.
        
        :param num_epochs: Number of epochs to train the model
        """
        for epoch in range(num_epochs):
            epoch_loss = self.train_epoch()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

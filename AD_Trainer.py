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
        self.latents = torch.nn.Parameter(torch.randn(len(self.dataloader.dataset), self.latent_dim).to(self.device))
        
        # Optimizer
        self.optimizer = optim.Adam(list(self.decoder.parameters()) + [self.latents], lr=lr)
        
        # Loss function
        self.loss = reconstruction_loss
        
    def train_epoch(self):
        self.decoder.train()
        running_loss = 0.0
        
        for batch_idx, (_, x) in enumerate(self.dataloader):
            images = x.to(self.device)
            batch_size = images.size(0)
            # print(f"batch size: {batch_size}")
            # Randomly initialize latent codes
            z = self.latents[batch_idx * batch_size : (batch_idx + 1) * batch_size, :]
            
            reconstructed_images = self.decoder(z)
            # print(f"Reconstructed image shape: {reconstructed_images.shape}")
            # print(f"image shape: {images.shape}")
            
            loss = self.loss(images, reconstructed_images)
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
            
            if best_loss is None or epoch_loss < best_loss:
                no_improvement = 0
                best_loss = epoch_loss
            else:
                no_improvement += 1
                if early_stopping is not None and no_improvement >= early_stopping:
                    break

        return losses

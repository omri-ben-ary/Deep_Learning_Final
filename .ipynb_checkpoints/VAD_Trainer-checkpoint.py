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

def elbo_loss(x, x_rec, mu, log_var):
    """
    :param x: the original images
    :param x_rec: the reconstructed images
    :return: the elbo loss
    """
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss(x, x_rec) + 0.0 * kl_loss
    

class VAD_Trainer:
    def __init__(self, var_decoder, dataloader, latent_dim=128, device='cpu', lr=1e-3):
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
        self.dataloader = dataloader
        self.device = device
        temp_latents = torch.randn(10, self.latent_dim).to(self.device)
        self.latents = torch.nn.Parameter(torch.stack([temp_latents[label,:] for label in dataloader.dataset.y])).to(device)

        # Optimizer
        self.optimizer = optim.Adam(list(self.var_decoder.parameters()) + [self.latents], lr=lr)
        
        # Loss function
        self.loss = elbo_loss
        
    def train_epoch(self):
        self.var_decoder.train()
        running_loss = 0.0
        
        for batch_idx, (_, x) in enumerate(self.dataloader):
            images = x.to(self.device)
            batch_size = images.size(0)
            z = self.latents[batch_idx * batch_size : (batch_idx + 1) * batch_size, :]
            reconstructed_images = self.var_decoder(z)
            
            loss = self.loss(images, reconstructed_images, self.var_decoder.mu, self.var_decoder.log_var)
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

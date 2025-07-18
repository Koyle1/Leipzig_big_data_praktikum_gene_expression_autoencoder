import torch
import torch.nn as nn
import torch.nn.functional as F

class CellVAE(nn.Module):
    """VAE Model Class with separate value and sparsity prediction heads"""
    def __init__(self, input_dim: int=512, latent_dim: int=10, use_variance: bool=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU()
        )
        self.z_mean = nn.Sequential(
            nn.Linear(in_features=64, out_features=latent_dim),
        )
        self.z_logvar = nn.Sequential(
            nn.Linear(in_features=64, out_features=latent_dim),
        ) if use_variance else None
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=input_dim),
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
    
    def encode(self, x):
        encoded = self.encoder(x)
        mu = self.z_mean(encoded)
        logvar = self.z_logvar(encoded) if self.z_logvar is not None else None
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if logvar is None:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z) 
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def loss_function(
        self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0
    ):
        """
        ELBO loss using RMSE + beta * KLD with normalization.
        """
        batch_size = x.size(0)
        
        if logvar is None:
            logvar = torch.ones_like(mu)

        MSE = F.mse_loss(recon_x, x, reduction='mean')
        RMSE = torch.sqrt(MSE)

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        return RMSE + beta * KLD, RMSE, KLD
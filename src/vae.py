import torch
import torch.nn as nn
import torch.nn.functional as F

class CellVAE(nn.Module):
    def __init__(self, input_dim: int=512, latent_dim: int=10, use_variance: bool=False):
        super().__init__()
        
         # Encoder network: maps input to a hidden representation
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU()
        )

        # Mean vector of the latent distribution
        self.z_mean = nn.Sequential(
            nn.Linear(in_features=64, out_features=latent_dim),
        )

        # Log-variance vector of the latent distribution (optional)
        self.z_logvar = nn.Sequential(
            nn.Linear(in_features=64, out_features=latent_dim),
        ) if use_variance else None

        # Decoder network: reconstructs input from latent representation
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=input_dim),
        )
        
        # Xavier initialization for weights, and bias initialized to small positive value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
    
    def encode(self, x):
        """
            Encode input into latent parameters (mu and logvar).
        """
        
        encoded = self.encoder(x)
        mu = self.z_mean(encoded)
        logvar = self.z_logvar(encoded) if self.z_logvar is not None else None
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
            Reparameterization trick: z = mu + std * eps
            Allows gradient flow through stochastic sampling.
        """
        
        if logvar is None:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
            Decode latent vector into reconstructed input.
        """
        
        return self.decoder(z) 
    
    def forward(self, x):
        """
            Full forward pass: encode -> sample -> decode.
        """
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def loss_function(
        self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0
    ):
        """
            ELBO loss = RMSE + beta * KLD.
            Uses RMSE instead of negative log-likelihood for reconstruction loss.
        """
        batch_size = x.size(0)
        
        if logvar is None:
            logvar = torch.ones_like(mu)

        MSE = F.mse_loss(recon_x, x, reduction='mean')
        RMSE = torch.sqrt(MSE)

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        return RMSE + beta * KLD, RMSE, KLD
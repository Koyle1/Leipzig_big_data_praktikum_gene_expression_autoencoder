import torch
from torch import nn, optim
from torch.nn import functional as F


class CellVAE(nn.Module):
    """VAE Model Class
    """
    def __init__(self, input_dim: int=512, latent_dim: int=10, use_variance: bool=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=input_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU()
        )
        self.z_mean = nn.Sequential(
            nn.Linear(in_features=32, out_features=latent_dim),
            nn.ReLU()
        )
        self.z_logvar = nn.Sequential(
            nn.Linear(in_features=32, out_features=latent_dim),
            nn.Softplus()
        ) if use_variance else None
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=input_dim),
            nn.Sigmoid()
        )
    def encode(self, x: torch.Tensor):
        encoding_raw = self.encoder.forward(x)
        mu = self.z_mean.forward(encoding_raw)
        if self.z_logvar is not None:
            logvar = self.z_logvar.forward(encoding_raw)
        else:
            logvar = None
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor=None):
        """Implementation of the "reparameterization trick".
        """
        if logvar is None:
            logvar = 1.0

        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(mu)

        return mu + epsilon * std
        

    def decode(self, x: torch.Tensor):
        return self.decoder.forward(x)

    def gumbel_softmax(self, x: torch.Tensor):
        pass

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = CellVAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
device = torch.device("cpu")
train_loader = torch

print(f"Using device: {device}")

def elbo_loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float=1.0):
    if logvar is None:
        logvar = torch.ones_like(mu)
    
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta * KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        recon_batch, mu, logvar = model(data)
        loss = elbo_loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # TODO: Log


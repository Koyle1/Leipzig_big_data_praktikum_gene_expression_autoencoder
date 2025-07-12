import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.nn.functional

class CellVAE(nn.Module):
    """VAE Model Class
    """
    def __init__(self, input_dim: int=512, latent_dim: int=10, use_variance: bool=False):
        super().__init__()
        self.encoder = nn.Sequential(
            # nn.Dropout(0.5),
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)


    def encode(self, x: torch.Tensor):
        encoding_raw = self.encoder.forward(x)
        mu = self.z_mean.forward(encoding_raw)
        if self.z_logvar is not None:
            logvar = self.z_logvar.forward(encoding_raw)
            # numerical stability TODO does not appear to work
            # logvar = torch.clamp(logvar, 0.1, 20)
        else:
            logvar = None

        # numerical stability TODO does not appear to work
        # mu = torch.where(mu < 0.000001, torch.zeros_like(mu) + 0.1, mu)
        
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor=None):
        """Implementation of the "reparameterization trick".
        Allowing unit variance if desired.
        """
        if logvar is None:
            logvar = torch.ones_like(mu)

        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(mu)

        return mu + epsilon * std
    
    def gumbel_softmax_sample(self, logits, temperature, eps=1e-8):
        """
        Sample from Gumbel-Softmax distribution
        """
        # Sample from Gumbel(0, 1)
        uniform = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(uniform + eps) + eps)
        
        # Add Gumbel noise and apply softmax with temperature
        y = logits + gumbel
        return F.softmax(y / temperature, dim=-1)
    
    def learned_gumbel_dropout(self, reconstructed_expr, temperature: torch.Tensor):
        """
        Implement learned dropout using Gumbel-Softmax sampling
        
        Args:
            reconstructed_expr: Tensor of shape [batch_size, n_features] 
                            (output from decoder, sigmoid activated)
            temperature: Tensor of shape [batch_size, n_features] or scalar
                        (controls hardness of sampling)
        
        Returns:
            masked_expr: Reconstructed expression with learned dropout applied
            dropout_probs: The learned dropout probabilities for analysis
        """
        
        # Step 1: Convert reconstructed values to drop probabilities
        # Higher reconstructed values -> lower drop probability
        expr_x_drop = -reconstructed_expr.pow(2)  # -x^2
        expr_x_drop_p = torch.exp(expr_x_drop)    # exp(-x^2), values in [0,1]
        
        # Step 2: Create complementary probabilities
        expr_x_keep_p = 1.0 - expr_x_drop_p
        
        # Step 3: Convert to log-probabilities (for numerical stability)
        eps = 1e-20
        log_drop_p = torch.log(expr_x_drop_p + eps)
        log_keep_p = torch.log(expr_x_keep_p + eps)
        
        # Step 4: Create logits for categorical distribution
        # Shape: [batch_size, n_features, 2]
        logits = torch.stack([log_drop_p, log_keep_p], dim=-1)
        
        # Step 5: Ensure temperature has correct shape
        if temperature.dim() == 1:
            # If temperature is per-feature, expand to match logits
            temperature = temperature.unsqueeze(0).unsqueeze(-1)  # [1, n_features, 1]
            temperature = temperature.expand_as(logits)  # [batch_size, n_features, 2]
        elif temperature.dim() == 0:
            # If scalar temperature, just use as-is
            pass
        
        # Step 6: Gumbel-Softmax sampling
        # Returns soft categorical samples [batch_size, n_features, 2]
        gumbel_samples = self.gumbel_softmax_sample(logits, temperature)
        
        # Step 7: Extract "keep" probabilities (index 1)
        keep_mask = gumbel_samples[:, :, 1]  # [batch_size, n_features]
        
        # Step 8: Apply learned dropout
        masked_expr = reconstructed_expr * keep_mask
        
        return masked_expr, expr_x_drop_p
    
    def decode(self, x: torch.Tensor, temperature: torch.Tensor):
        reconstructed = self.decoder.forward(x)

        # Apply learned Gumbel-Softmax dropout
        # masked_output, dropout_probs = self.learned_gumbel_dropout(reconstructed, temperature)
        
        return reconstructed, None

    def forward(self, x, temperature: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        self.last_z = z.detach()
        return self.decode(z, temperature), mu, logvar

def elbo_loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float=1.0):
    """Uses mean reduction as opposed to summing in the original VAE paper

    TODO: evaluate mean vs sum reduction
    """
    if logvar is None:
        logvar = torch.ones_like(mu)
    
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # free bits:
    # KLD_per_dim = -0.5 * 1 + logvar - mu.pow(2) - logvar.exp()
    # KLD = torch.sum(torch.maximum(KLD_per_dim, torch.full_like(KLD_per_dim, 0.2)))

    return BCE + beta * KLD, BCE, KLD

def elbo_loss_function_normalized(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float=1.0):
    """Uses mean reduction as opposed to summing in the original VAE paper

    TODO: evaluate mean vs sum reduction
    """
    batch_size = x.size(0)

    if logvar is None:
        logvar = torch.ones_like(mu)
    
    # Reconstruction loss (normalized by input dimension)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.numel()
    
    # KL divergence (normalized by latent dimension and batch size)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1) / batch_size
    KLD = KLD.mean() 
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (batch_size * mu.size(1))
    
    return BCE + beta * KLD, BCE, KLD
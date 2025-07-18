import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryAveragePrecision

pr_auc_metric = BinaryAveragePrecision()

# Approach 1: Dual-head decoder (Recommended)
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
        
        # Shared decoder backbone
        self.decoder_backbone = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
        )
        
        # Separate heads for value and sparsity prediction
        self.value_head = nn.Linear(in_features=512, out_features=input_dim)
        self.sparsity_head = nn.Linear(in_features=512, out_features=input_dim)
        
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
        # Shared decoder processing
        decoded_features = self.decoder_backbone(z)
        
        # Separate predictions
        values = self.value_head(decoded_features)
        sparsity_logits = self.sparsity_head(decoded_features)
        
        # Combine: values weighted by sparsity probability
        sparsity_probs = torch.sigmoid(sparsity_logits)
        reconstruction = values * sparsity_probs
        
        return reconstruction, values, sparsity_logits
    
    def forward(self, x, tau0=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction, values, sparsity_logits = self.decode(z)
        return reconstruction, mu, logvar, values, sparsity_logits
    
    from torchmetrics.classification import BinaryAveragePrecision


    def loss_function(self, recon_x, x, mu, logvar, values=None, sparsity_logits=None,
                  beta=0.0, sparsity_threshold=1e-4, 
                  sparsity_weight=1.0, value_weight=1.0, fn_penalty=5.0):
            """
            Loss function for dual-head architecture with strong false negative avoidance.
            Includes PR-AUC metric for sparsity prediction.
            """
        
            if logvar is None:
                logvar = torch.ones_like(mu)
        
            batch_size = x.size(0)
        
            # Binary labels for sparsity
            is_nonzero = (torch.abs(x) > sparsity_threshold).float()
            #is_nonzero_long = (torch.abs(x) > sparsity_threshold).long()
        
            # Method 1: Weighted BCE
            num_zeros = (is_nonzero == 0).sum()
            num_nonzeros = (is_nonzero == 1).sum()
        
            if num_nonzeros > 0:
                pos_weight = (num_zeros / num_nonzeros) * fn_penalty
            else:
                pos_weight = fn_penalty
        
            sparsity_loss = F.binary_cross_entropy_with_logits(
                sparsity_logits, is_nonzero,
                pos_weight=pos_weight,
                reduction='mean'
            )
        
            # Method 2: Explicit FN penalty
            sparsity_probs = torch.sigmoid(sparsity_logits)
            pred_nonzero = (sparsity_probs > 0.5).float()
        
            false_negatives = ((pred_nonzero == 0) & (is_nonzero == 1)).float()
            fn_penalty_loss = false_negatives.mean() * fn_penalty
        
            # Method 3: Recall penalty
            true_positives = ((pred_nonzero == 1) & (is_nonzero == 1)).sum()
            actual_positives = (is_nonzero == 1).sum()
        
            if actual_positives > 0:
                recall = true_positives / actual_positives
                recall_loss = (1.0 - recall) * fn_penalty
            else:
                recall_loss = torch.tensor(0.0, device=x.device)
        
            # Combined sparsity loss
            total_sparsity_loss = sparsity_loss + fn_penalty_loss + recall_loss
        
            # Value RMSE (non-zero targets only)
            nonzero_mask = is_nonzero > 0
            if nonzero_mask.sum() > 0:
                nonzero_recon = values[nonzero_mask]
                nonzero_true = x[nonzero_mask]
                value_mse = F.mse_loss(nonzero_recon, nonzero_true, reduction='mean')
                value_rmse = torch.sqrt(value_mse + 1e-8)
            else:
                value_rmse = torch.tensor(0.0, device=x.device, requires_grad=True)
        
            # KL divergence
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
            # Composite loss
            total_loss = (value_weight * value_rmse +
                          sparsity_weight * total_sparsity_loss +
                          beta * KLD)
        
            # Metrics
            sparsity_ratio = torch.sum(is_nonzero) / (batch_size * x.size(-1))
            sparsity_accuracy = ((pred_nonzero == is_nonzero).float().mean())
        
            # PR-AUC: torchmetrics expects 1D inputs
            #pr_auc = pr_auc_metric(sparsity_probs.view(-1),is_nonzero_long.view(-1))

            pr_auc = 0
        
            return total_loss, value_rmse, total_sparsity_loss, KLD, pr_auc

    
    def legacy_loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 0.0):
        """
        ELBO loss using RMSE + beta * KLD.
        """
        if logvar is None:
            logvar = torch.ones_like(mu)

        mse = F.mse_loss(recon_x, x, reduction='mean')
        rmse = torch.sqrt(mse)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return rmse + beta * KLD, rmse, KLD

    def loss_function_normalized(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0):
        """
        ELBO loss using binary cross entropy + beta * KLD with normalization.
        """
        batch_size = x.size(0)
        if logvar is None:
            logvar = torch.ones_like(mu)

        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.numel()
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1) / batch_size
        KLD = KLD.mean()
        return BCE + beta * KLD, BCE, KLD
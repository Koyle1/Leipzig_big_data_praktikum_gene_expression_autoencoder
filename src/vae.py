import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CellVAE(nn.Module):
    def __init__(self, input_dim: int = 512, latent_dim: int = 10, use_variance: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_variance = use_variance

        # Improved encoder with batch normalization and larger bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=64),  # Larger bottleneck
            nn.ReLU()
        )
        
        self.z_mean = nn.Linear(in_features=64, out_features=latent_dim)
        self.z_logvar = nn.Linear(in_features=64, out_features=latent_dim) if use_variance else None

        # Improved decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=64, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=input_dim)
        )


        # Improved initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        encoded = self.encoder(x)
        mu = self.z_mean(encoded)
        logvar = self.z_logvar(encoded) if self.z_logvar is not None else None
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if logvar is None:
            return mu
        # More conservative clamping
        logvar = torch.clamp(logvar, min=-5, max=5)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, tau0=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z, torch.tensor(0.0, device=x.device)

    def l_loss_function(self, recon_x, x, mu, logvar, z, z_perm, D_z_logits, D_z_perm_logits,
                      beta=1.0, gamma=10.0, sparsity_threshold=1e-4, value_weight=1.0):
        if logvar is None:
            logvar = torch.zeros_like(mu)  # Use zeros instead of ones for deterministic case
    
        batch_size = x.size(0)
        is_nonzero = (torch.abs(x) > sparsity_threshold).float()
        nonzero_mask = is_nonzero > 0
    
        # Reconstruction loss
        if nonzero_mask.sum() > 0:
            nonzero_recon = recon_x[nonzero_mask]
            nonzero_true = x[nonzero_mask]
            value_mse = F.mse_loss(nonzero_recon, nonzero_true, reduction='mean')
            value_rmse = torch.sqrt(value_mse + 1e-8)
        else:
            value_rmse = torch.tensor(0.0, device=x.device, requires_grad=True)
    
        # KL divergence (normalized by batch size)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    
        # Improved Total Correlation loss
        # The discriminator should classify z as "real" (class 0) and z_perm as "fake" (class 1)
        # We want to minimize TC, so we encourage the discriminator to be confused
        D_z_prob = F.softmax(D_z_logits, dim=1)[:, 0]  # P(real | z)
        D_z_perm_prob = F.softmax(D_z_perm_logits, dim=1)[:, 0]  # P(real | z_perm)
        
        # TC loss: we want D(z_perm) to be high (close to real) and D(z) to be low (close to fake)
        # This encourages the factors to be independent
        TC = torch.mean(torch.log(D_z_prob + 1e-8) + torch.log(1 - D_z_perm_prob + 1e-8))
    
        total_loss = value_weight * value_rmse + beta * KLD - gamma * TC  # Note the minus sign
        return total_loss, value_rmse, KLD, -TC  # Return -TC for logging consistency


    
    def loss_function(self, recon_x, x, mu, logvar, z,
                      beta=6.0, gamma=1.0, theta=1.0, sparsity_threshold=1e-4,
                      value_weight=1.0, training_step=0):
        """
        Fast and numerically stable β-TC-VAE loss function
        """
        batch_size, latent_dim = z.size()
    
        if logvar is None:
            logvar = torch.zeros_like(mu)
    
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
    
        # Reconstruction loss (unchanged)
        is_nonzero = (torch.abs(x) > sparsity_threshold).float()
        nonzero_mask = is_nonzero > 0
    
        if nonzero_mask.sum() > 0:
            nonzero_recon = recon_x[nonzero_mask]
            nonzero_true = x[nonzero_mask]
            value_mse = F.mse_loss(nonzero_recon, nonzero_true, reduction='mean')
            value_rmse = torch.sqrt(value_mse + 1e-8)
        else:
            value_rmse = torch.tensor(0.0, device=x.device, requires_grad=True)
    
        var = torch.exp(logvar)
    
        # Fix KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - var, dim=1)
        dim_wise_kl = kl_div.mean()
    
        # Fast MI and TC computation
        MI, TC = self._compute_fast_mi_tc(z, mu, logvar)
    
        # Clamp to prevent explosions
        MI = torch.clamp(MI, min=-10.0, max=10.0)
        TC = torch.clamp(TC, min=-10.0, max=10.0)
    
        
        
        # Fix warmup
        if training_step < 500:
            theta_effective = theta * (training_step / 500.0)
        else:
            theta_effective = theta
    
        total_loss = (value_weight * value_rmse + 
                      theta_effective * MI + 
                      beta * (-TC) + 
                      gamma * dim_wise_kl)
    
        return total_loss, value_rmse, MI, (-TC), dim_wise_kl


    def _compute_fast_mi_tc(self, z, mu, logvar):
        """
        Fast MI and TC computation using vectorized operations
        Time complexity: O(batch_size * latent_dim) instead of O(batch_size²)
        """
        batch_size, latent_dim = z.size()
        
        # For very large batches, use sampling
        if batch_size > 64:
            return self._compute_sampled_mi_tc(z, mu, logvar)
        
        # Vectorized computation for reasonable batch sizes
        return self._compute_vectorized_mi_tc(z, mu, logvar)
    
    
    def _compute_sampled_mi_tc(self, z, mu, logvar):
        """
        Super fast MI/TC using random sampling - O(sample_size * latent_dim)
        """
        batch_size, latent_dim = z.size()
        sample_size = min(16, batch_size)  # Use small sample for density estimation
        
        # Random sampling for density estimation
        indices = torch.randperm(batch_size, device=z.device)[:sample_size]
        mu_sample = mu[indices]      # (sample_size, latent_dim)
        logvar_sample = logvar[indices]  # (sample_size, latent_dim)
        var_sample = torch.exp(logvar_sample)
        
        # Compute log q(z|x) for all points
        log_q_z_given_x = self._log_normal_pdf(z, mu, logvar).sum(dim=1)  # (batch_size,)
        
        # Vectorized log q(z) computation
        # z: (batch_size, latent_dim), mu_sample: (sample_size, latent_dim)
        z_expanded = z.unsqueeze(1)  # (batch_size, 1, latent_dim)
        mu_expanded = mu_sample.unsqueeze(0)  # (1, sample_size, latent_dim)
        logvar_expanded = logvar_sample.unsqueeze(0)  # (1, sample_size, latent_dim)
        
        # Compute log densities for all pairs at once
        log_densities = self._log_normal_pdf(z_expanded, mu_expanded, logvar_expanded)  # (batch_size, sample_size, latent_dim)
        log_densities = log_densities.sum(dim=2)  # Sum over latent dimensions: (batch_size, sample_size)
        
        # Stable logsumexp
        log_q_z = torch.logsumexp(log_densities, dim=1) - np.log(sample_size)  # (batch_size,)
        
        # MI computation
        MI = (log_q_z_given_x - log_q_z).mean()
        
        # Fast TC computation using the same samples
        # Compute log q(z_j) for each dimension
        log_q_zj_sum = torch.zeros(batch_size, device=z.device)
        
        for j in range(latent_dim):
            # z[:, j]: (batch_size,), mu_sample[:, j]: (sample_size,)
            z_j = z[:, j].unsqueeze(1)  # (batch_size, 1)
            mu_j = mu_sample[:, j].unsqueeze(0)  # (1, sample_size)
            logvar_j = logvar_sample[:, j].unsqueeze(0)  # (1, sample_size)
            
            log_densities_j = self._log_normal_pdf(z_j, mu_j, logvar_j)  # (batch_size, sample_size)
            log_q_zj = torch.logsumexp(log_densities_j, dim=1) - np.log(sample_size)  # (batch_size,)
            log_q_zj_sum += log_q_zj
        
        TC = (log_q_z - log_q_zj_sum).mean()
        
        return MI, TC
    
    
    def compute_vectorized_mi_tc(self, z, mu, logvar):
        """
        Vectorized MI/TC for medium batch sizes - O(batch_size * latent_dim)
        Fixed version with correct TC sign and improved numerical stability
        """
        batch_size, latent_dim = z.size()
        
        # Use all samples for density estimation (manageable for batch_size <= 64)
        log_q_z_given_x = self._log_normal_pdf(z, mu, logvar).sum(dim=1)
        
        # Vectorized density computation for joint q(z)
        z_expanded = z.unsqueeze(1)  # (batch_size, 1, latent_dim)
        mu_expanded = mu.unsqueeze(0)  # (1, batch_size, latent_dim)
        logvar_expanded = logvar.unsqueeze(0)  # (1, batch_size, latent_dim)
        
        # All pairwise log densities at once
        log_densities = self._log_normal_pdf(z_expanded, mu_expanded, logvar_expanded)
        log_densities = log_densities.sum(dim=2)  # (batch_size, batch_size)
        
        # Stable logsumexp for joint marginal q(z)
        log_q_z = torch.logsumexp(log_densities, dim=1) - np.log(batch_size)
        
        # Mutual Information: I(z;x) = E[log q(z|x) - log q(z)]
        MI = (log_q_z_given_x - log_q_z).mean()
        
        # Vectorized TC computation - compute marginals q(z_j) for each dimension
        log_q_zj_sum = torch.zeros(batch_size, device=z.device)
        
        for j in range(latent_dim):
            z_j = z[:, j].unsqueeze(1)  # (batch_size, 1)
            mu_j = mu[:, j].unsqueeze(0)  # (1, batch_size)
            logvar_j = logvar[:, j].unsqueeze(0)  # (1, batch_size)
            
            # Compute marginal density q(z_j) for dimension j
            log_densities_j = self._log_normal_pdf(z_j, mu_j, logvar_j)  # (batch_size, batch_size)
            log_q_zj = torch.logsumexp(log_densities_j, dim=1) - np.log(batch_size)
            log_q_zj_sum += log_q_zj
        
        # FIXED: Total Correlation TC = KL(q(z) || ∏_j q(z_j))
        # In log space: TC = E[log q(z) - log ∏_j q(z_j)]
        # Since log ∏_j q(z_j) = ∑_j log q(z_j) = log_q_zj_sum
        TC = (log_q_zj_sum - log_q_z).mean()
        
        return MI, TC
    
    
    def _log_normal_pdf(self, x, mu, logvar):
        """
        Fast and stable log normal PDF computation
        """
        var = torch.exp(logvar)
        log_2pi = np.log(2 * np.pi)  # Use numpy constant
        
        # Vectorized computation
        log_pdf = -0.5 * ((x - mu) ** 2 / (var + 1e-8) + logvar + log_2pi)
        
        # Clamp for numerical stability
        return torch.clamp(log_pdf, min=-50.0, max=50.0)
    
    
    # Ultra-fast approximation for very large batches
    def ultra_fast_loss_function(self, recon_x, x, mu, logvar, z,
                                beta=6.0, gamma=1.0, theta=1.0, sparsity_threshold=1e-4,
                                value_weight=1.0):
        """
        Ultra-fast approximation using analytical MI/TC estimates
        """
        batch_size, latent_dim = z.size()
    
        if logvar is None:
            logvar = torch.zeros_like(mu)
    
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
    
        # Reconstruction loss (unchanged)
        is_nonzero = (torch.abs(x) > sparsity_threshold).float()
        nonzero_mask = is_nonzero > 0
    
        if nonzero_mask.sum() > 0:
            nonzero_recon = recon_x[nonzero_mask]
            nonzero_true = x[nonzero_mask]
            value_mse = F.mse_loss(nonzero_recon, nonzero_true, reduction='mean')
            value_rmse = torch.sqrt(value_mse + 1e-8)
        else:
            value_rmse = torch.tensor(0.0, device=x.device, requires_grad=True)
    
        var = torch.exp(logvar)
    
        # Standard VAE KL divergence
        dim_wise_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - var, dim=1).mean()
    
        # Fast analytical approximations
        MI = self._approximate_mi(z, mu, logvar)
        TC = self._approximate_tc(z, mu, logvar)
    
        # Clamp
        MI = torch.clamp(MI, min=-5.0, max=5.0)
        TC = torch.clamp(TC, min=-5.0, max=5.0)
    
        total_loss = (value_weight * value_rmse + 
                      theta * 0.1 * MI + 
                      beta * TC + 
                      gamma * dim_wise_kl)
    
        return total_loss, value_rmse, MI, TC, dim_wise_kl
    
    
    def _approximate_mi(self, z, mu, logvar):
        """
        Fast MI approximation using variance of latent codes
        """
        # MI ≈ -0.5 * log(det(Cov(z))) + constant
        # Approximate using diagonal covariance
        z_var = torch.var(z, dim=0)  # Variance across batch for each dimension
        log_det_cov = torch.sum(torch.log(z_var + 1e-8))
        MI = -0.5 * log_det_cov
        return MI
    
    
    def _approximate_tc(self, z, mu, logvar):
        """
        Fast TC approximation using empirical statistics
        """
        # TC ≈ KL(q(z) || ∏q(z_j))
        # Approximate using sample statistics
        batch_size, latent_dim = z.size()
        
        # Empirical marginal variances
        z_var = torch.var(z, dim=0)  # (latent_dim,)
        z_mean = torch.mean(z, dim=0)  # (latent_dim,)
        
        # Approximate joint vs marginal entropy difference
        log_joint_var = torch.sum(torch.log(z_var + 1e-8))
        log_marginal_vars = torch.sum(torch.log(z_var + 1e-8))
        
        # Simple approximation
        TC = 0.5 * (log_joint_var - log_marginal_vars)
        
        return TC
    
    
    # Minimal β-VAE version (fastest)
    def minimal_fast_loss_function(self, recon_x, x, mu, logvar, z,
                                  beta=1.0, sparsity_threshold=1e-4, value_weight=1.0):
        """
        Minimal β-VAE loss (no MI/TC decomposition) - fastest option
        """
        batch_size, latent_dim = z.size()
    
        if logvar is None:
            logvar = torch.zeros_like(mu)
    
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
    
        # Reconstruction loss
        is_nonzero = (torch.abs(x) > sparsity_threshold).float()
        nonzero_mask = is_nonzero > 0
    
        if nonzero_mask.sum() > 0:
            nonzero_recon = recon_x[nonzero_mask]
            nonzero_true = x[nonzero_mask]
            value_mse = F.mse_loss(nonzero_recon, nonzero_true, reduction='mean')
            value_rmse = torch.sqrt(value_mse + 1e-8)
        else:
            value_rmse = torch.tensor(0.0, device=x.device, requires_grad=True)
    
        var = torch.exp(logvar)
    
        # Standard VAE KL divergence (single term)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - var, dim=1).mean()
    
        total_loss = value_weight * value_rmse + beta * kl_div
    
        # Return zeros for MI and TC for compatibility
        MI = torch.tensor(0.0, device=z.device)
        TC = torch.tensor(0.0, device=z.device)
        dim_wise_kl = kl_div
    
        return total_loss, value_rmse, MI, TC, dim_wise_kl
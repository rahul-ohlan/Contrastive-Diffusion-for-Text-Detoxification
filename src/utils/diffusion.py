"""
Gaussian diffusion process for text embeddings.
"""
import torch
import numpy as np

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_schedule='linear', device=None):
        self.timesteps = timesteps
        
        # Define beta schedule
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-calculate diffusion parameters - keep on CPU
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.]), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1}) - keep on CPU
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0) - keep on CPU
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def add_noise(self, x_start, t, noise=None):
        """
        Add noise to embeddings at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # t should be on CPU, move results to x_start's device
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].to(x_start.device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(x_start.device)
        
        # Expand dimensions to match x_start
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1)
        
        noisy_embeddings = (
            sqrt_alphas_cumprod_t * x_start +
            sqrt_one_minus_alphas_cumprod_t * noise
        )
        
        return noisy_embeddings, noise
        
    @torch.no_grad()
    def p_sample(self, model, x_t, t, toxic_embeddings, toxic_mask=None):
        """
        Sample from p(x_{t-1} | x_t)
        """
        # Move t to GPU for model forward pass
        t_device = t.to(x_t.device)
        pred_noise = model(x_t, t_device, toxic_embeddings, toxic_mask)
        
        # Get parameters for posterior distribution - t should be on CPU
        alpha = self.alphas[t].to(x_t.device)
        alpha_cumprod = self.alphas_cumprod[t].to(x_t.device)
        beta = self.betas[t].to(x_t.device)
        
        # Expand dimensions
        alpha = alpha.view(-1, 1, 1)
        alpha_cumprod = alpha_cumprod.view(-1, 1, 1)
        beta = beta.view(-1, 1, 1)
        
        # Calculate mean
        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod) * pred_noise) / torch.sqrt(alpha_cumprod)
        mean = (
            (1 - alpha) * pred_x0 / torch.sqrt(1 - alpha_cumprod) +
            torch.sqrt(alpha) * x_t
        )
        
        # Add noise for variance
        noise = torch.randn_like(x_t)
        variance = torch.sqrt(beta) * noise
        
        return mean + variance
        
    @torch.no_grad()
    def p_sample_loop(self, model, shape, toxic_embeddings, toxic_mask=None):
        """
        Generate samples by iteratively denoising
        """
        device = next(model.parameters()).device
        x_t = torch.randn(shape, device=device)
        
        for t in reversed(range(self.timesteps)):
            # Create t_batch on CPU for indexing
            t_batch = torch.full((shape[0],), t, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t_batch, toxic_embeddings, toxic_mask)
            
        return x_t 
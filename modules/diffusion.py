"""Latent private diffusion module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        scale = torch.log(torch.tensor(10000.0, device=t.device)) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=t.device) * -scale)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class LatentPrivateDiffusion(nn.Module):
    def __init__(self, latent_dim: int, cond_dim: int, T: int = 20, time_dim: int = 32):
        super().__init__()
        self.T = T
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.denoiser = nn.Sequential(
            nn.Linear(latent_dim + cond_dim + time_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, latent_dim),
        )
        betas = torch.linspace(1e-4, 0.02, T)
        self.register_buffer('alpha_bar', torch.cumprod(1.0 - betas, dim=0))

    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        alpha_bar = self.alpha_bar[t].unsqueeze(-1)
        zt = alpha_bar.sqrt() * z0 + (1 - alpha_bar).sqrt() * noise
        return zt, noise

    def predict_noise(self, zt, t, cond):
        t_emb = self.time_embed(t.float())
        return self.denoiser(torch.cat([zt, cond, t_emb], dim=-1))

    def diffusion_loss(self, z0, cond):
        batch_size = z0.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=z0.device)
        zt, noise = self.q_sample(z0, t)
        pred = self.predict_noise(zt, t, cond)
        return F.mse_loss(pred, noise)

    def refine(self, z0, cond, t_value=None):
        if t_value is None:
            t_value = max(1, self.T // 2)
        t = torch.full((z0.shape[0],), min(t_value, self.T - 1), device=z0.device, dtype=torch.long)
        zt, _ = self.q_sample(z0, t)
        pred = self.predict_noise(zt, t, cond)
        alpha_bar = self.alpha_bar[t].unsqueeze(-1)
        return (zt - (1 - alpha_bar).sqrt() * pred) / (alpha_bar.sqrt() + 1e-8)


__all__ = ["SinusoidalTimeEmbedding", "LatentPrivateDiffusion"]

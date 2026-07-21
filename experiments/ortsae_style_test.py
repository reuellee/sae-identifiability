
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import itertools
import time
import os

# --- Configuration ---
class Config:
    # Model dimensions
    d = 64  # Ambient dimension
    m = 32  # Number of SAE latents

    # Data generation
    n_background_features = 30
    bg_feature_rate = 0.08
    parent_solo_prob = 0.2
    joint_prob = 0.2
    # child_solo_prob 'eps' is swept

    # SAE hyperparameters
    l1_lambda = 0.2
    # ortho_gamma is swept

    # Training parameters
    n_steps = 15000
    batch_size = 2048
    lr = 1e-3
    
    # Experiment sweeps
    epsilons = [0.005, 0.01, 0.02, 0.03]
    gammas = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    n_seeds = 8

    # OrtSAE penalty
    K_chunks = 4 # 32 latents / 4 chunks = 8 latents per chunk

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Toy Data Generation ---
def generate_toy_data(cfg, parent_child_features, eps_tensor):
    """Generates a batch of data for all runs, with run-specific epsilons."""
    n_runs = eps_tensor.shape[0]
    
    # 1. Background features
    bg_coeffs = torch.rand(n_runs, cfg.batch_size, cfg.n_background_features, device=cfg.device) * 0.4 + 0.8
    bg_active = torch.rand(n_runs, cfg.batch_size, cfg.n_background_features, device=cfg.device) < cfg.bg_feature_rate
    bg_activations = bg_coeffs * bg_active
    
    # 2. Parent-child features
    ap, ac = parent_child_features
    pc_activations = torch.zeros(n_runs, cfg.batch_size, 2, device=cfg.device)
    
    rand_case = torch.rand(n_runs, cfg.batch_size, device=cfg.device)
    
    # Reshape eps for broadcasting
    eps_broadcast = eps_tensor.view(n_runs, 1)

    p0 = cfg.joint_prob
    p1 = p0 + cfg.parent_solo_prob
    p2 = p1 + eps_broadcast

    is_joint = rand_case < p0
    is_parent_solo = (rand_case >= p0) & (rand_case < p1)
    is_child_solo = (rand_case >= p1) & (rand_case < p2)

    # Use broadcasting to safely set activations
    pc_activations += is_joint.unsqueeze(-1) * torch.tensor([1.0, 1.0], device=cfg.device)
    pc_activations += is_parent_solo.unsqueeze(-1) * torch.tensor([1.0, 0.0], device=cfg.device)
    pc_activations += is_child_solo.unsqueeze(-1) * torch.tensor([0.0, 1.0], device=cfg.device)

    # Combine all features
    all_feature_activations = torch.cat([bg_activations, pc_activations], dim=2)
    
    # Project to ambient space
    data = torch.einsum('rbf,fd->rbd', all_feature_activations, cfg.ground_truth_features)
    
    return data

# --- SAE Model ---
class BatchedSAE(nn.Module):
    def __init__(self, cfg, n_runs):
        super().__init__()
        self.cfg = cfg
        self.n_runs = n_runs

        self.W_e = nn.Parameter(torch.randn(n_runs, cfg.m, cfg.d, device=cfg.device) / np.sqrt(cfg.d))
        self.b_e = nn.Parameter(torch.zeros(n_runs, cfg.m, device=cfg.device))
        self.D = nn.Parameter(torch.randn(n_runs, cfg.d, cfg.m, device=cfg.device))
        with torch.no_grad():
            self.D.div_(torch.norm(self.D, dim=1, keepdim=True))

    def forward(self, x):
        # x: (n_runs, batch_size, d)
        c_pre = torch.einsum('rmd,rbd->rbm', self.W_e, x) + self.b_e.unsqueeze(1)
        c = torch.relu(c_pre)
        x_hat = torch.einsum('rdm,rbm->rbd', self.D, c)
        return x_hat, c

    def normalize_decoder(self):
        with torch.no_grad():
            self.D.div_(torch.norm(self.D, dim=1, keepdim=True))

# --- Orthogonality Penalty ---
def ortsae_penalty(D, K_chunks):
    # D: (n_runs, d, m)
    n_runs, d, m = D.shape
    chunk_size = m // K_chunks
    
    # Normalize decoders for cosine similarity calculation
    D_norm = D / torch.norm(D, dim=1, keepdim=True)
    
    # Reshape for chunked processing
    D_chunked = D_norm.transpose(1, 2).reshape(n_runs, K_chunks, chunk_size, d)
    
    # Calculate pairwise cosine similarities within each chunk
    # (r, k, c, d) x (r, k, z, d) -> (r, k, c, z)
    cos_sims = torch.einsum('rkcd,rkzd->rkcz', D_chunked, D_chunked)
    
    # Set diagonal to a very low value to ignore self-similarity
    cos_sims.diagonal(dim1=-2, dim2=-1).fill_(-1e9)
    
    # Find max cosine similarity for each vector in each chunk
    max_cos_sim, _ = cos_sims.max(dim=-1)
    
    # Square it, average within chunks, then average across chunks
    penalty = (max_cos_sim**2).mean(dim=-1).mean(dim=-1)
    
    return penalty # shape: (n_runs,)


# --- Analysis ---
def analyze_results(sae, cfg, parent_child_features, run_params):
    ap, ac = parent_child_features
    
    # Create probe inputs: a_p and a_c
    probes = torch.stack([ap, ac], dim=0).to(cfg.device) # (2, d)
    
    # Expand probes for all runs
    probes_batched = probes.unsqueeze(0).expand(sae.n_runs, -1, -1) # (n_runs, 2, d)
    
    # Get encoder responses to probes
    # (n_runs, m, d) x (n_runs, d, 2) -> (n_runs, m, 2)
    encoder_responses = torch.einsum('rmd,rdp->rmp', sae.W_e.detach(), probes_batched.transpose(1, 2))
    
    # Find the latent with max response to child-solo probe (a_c)
    # This is our "child-side latent"
    child_latent_indices = torch.argmax(encoder_responses[:, :, 1], dim=1) # (n_runs,)
    
    # Get the decoder vectors for these latents
    child_decoder_vectors = sae.D[torch.arange(sae.n_runs), :, child_latent_indices] # (n_runs, d)
    
    # Project these vectors onto the (ap, ac) plane
    proj_ap = torch.einsum('rd,d->r', child_decoder_vectors, ap)
    proj_ac = torch.einsum('rd,d->r', child_decoder_vectors, ac)
    
    # Calculate angle in the plane
    angle_rad = torch.atan2(proj_ac, proj_ap)
    angle_deg = torch.rad2deg(angle_rad) # Absorbed ~45 deg, Faithful ~90 deg

    # Also find top K latents in the plane to identify anti-rotation
    # Project all decoder vectors onto the plane
    D_proj_ap = torch.einsum('rdm,d->rm', sae.D.detach(), ap)
    D_proj_ac = torch.einsum('rdm,d->rm', sae.D.detach(), ac)
    
    rhos = torch.sqrt(D_proj_ap**2 + D_proj_ac**2)
    phis = torch.rad2deg(torch.atan2(D_proj_ac, D_proj_ap))

    # Get top 5 latents by rho (in-plane magnitude)
    top_k_rhos, top_k_indices = torch.topk(rhos, 5, dim=1)

    results = []
    for i in range(sae.n_runs):
        res = run_params[i].copy()
        res['child_latent_angle_deg'] = angle_deg[i].item()
        
        # Store top 5 rhos/phis
        for j in range(5):
            idx = top_k_indices[i, j].item()
            res[f'top_{j+1}_rho'] = top_k_rhos[i, j].item()
            res[f'top_{j+1}_phi_deg'] = phis[i, idx].item()

        results.append(res)
        
    return pd.DataFrame(results)

# --- Main Training ---
def main():
    cfg = Config()
    print(f"Using device: {cfg.device}")
    
    # Create parameter combinations
    run_params = []
    for gamma, eps, seed in itertools.product(cfg.gammas, cfg.epsilons, range(cfg.n_seeds)):
        run_params.append({'gamma': gamma, 'eps': eps, 'seed': seed})
    n_runs = len(run_params)
    print(f"Total experiment runs: {n_runs}")

    # Set all seeds
    # This is not perfect for batched runs but provides some initial diversity
    torch.manual_seed(0)

    # Setup ground truth features
    cfg.ground_truth_features = torch.randn(cfg.n_background_features + 2, cfg.d, device=cfg.device)
    cfg.ground_truth_features, _ = torch.linalg.qr(cfg.ground_truth_features.T)
    cfg.ground_truth_features = cfg.ground_truth_features.T
    
    parent_child_features = cfg.ground_truth_features[-2:, :]
    ap, ac = parent_child_features[0], parent_child_features[1]

    # Initialize model and optimizer
    sae = BatchedSAE(cfg, n_runs).to(cfg.device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    # Prepare broadcast-able hyperparameter tensors
    gammas_tensor = torch.tensor([p['gamma'] for p in run_params], device=cfg.device)
    eps_tensor = torch.tensor([p['eps'] for p in run_params], device=cfg.device)

    start_time = time.time()
    for step in range(cfg.n_steps):
        sae.train()
        
        # Generate data vectorized across all runs
        x = generate_toy_data(cfg, parent_child_features, eps_tensor)

        optimizer.zero_grad()
        
        x_hat, c = sae(x)
        
        # Losses
        recon_loss = ((x - x_hat)**2).sum(-1).mean(-1) # SUM over d, mean over batch (matches validated recipe)
        l1_loss = torch.norm(c, p=1, dim=-1).mean(dim=-1) # Avg over batch
        ortho_loss = ortsae_penalty(sae.D, cfg.K_chunks)
        
        total_loss = recon_loss + cfg.l1_lambda * l1_loss + gammas_tensor * ortho_loss
        
        # Backward pass on the sum of losses for all runs
        total_loss.sum().backward()
        optimizer.step()
        
        # Normalize decoder weights
        sae.normalize_decoder()

        # LR decay at halfway (validated recipe)
        if step == cfg.n_steps // 2:
            for g in optimizer.param_groups:
                g['lr'] = cfg.lr / 3
        
        if (step + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step+1}/{cfg.n_steps} | Avg Loss: {total_loss.mean().item():.4f} | Time: {elapsed:.1f}s")

    print("Training finished.")
    
    # --- Analysis and Save ---
    print("Analyzing results...")
    sae.eval()
    results_df = analyze_results(sae, cfg, parent_child_features, run_params)
    
    output_path = 'experiment_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from sklearn.model_selection import train_test_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


l_min = 520.0
l_max = 640.0
w_stroke = 0.0

# ================== Delta geometry ==================
def load_geometry_KR3D1200() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回:
      base_points:   [3,3] (mm)
      plat_local:    [3,3]
    """
    Rb, Rp = 280.0, 80.0
    theta = np.deg2rad([0, 120, 240])
    z_base = 0.0

    B = np.stack([
        Rb*np.cos(theta),
        Rb*np.sin(theta),
        np.full_like(theta, z_base)
    ], axis=1).astype(np.float32)        # (3,3)

    P_local = np.stack([
        Rp*np.cos(theta),
        Rp*np.sin(theta),
        np.zeros_like(theta)
    ], axis=1).astype(np.float32)        # (3,3)

    P_local -= P_local.mean(axis=0, keepdims=True)

    base_points = torch.from_numpy(B).to(device)
    plat_local  = torch.from_numpy(P_local).to(device)
    return base_points, plat_local

BASE_POINTS, PLAT_LOCAL = load_geometry_KR3D1200()  # [3,3], [3,3]

class DeltaPMP(nn.Module):
    def __init__(self, platform_points_local: torch.Tensor, num_legs: int = 3, hidden_dim: int = 256):
        super().__init__()
        assert platform_points_local.shape == (num_legs, 3)
        self.num_legs = num_legs
        self.register_buffer('platform_points_local', platform_points_local.clone())  # [3,3]

        self.fc_output = nn.Sequential(
            nn.Linear(num_legs * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_legs),
            nn.Softplus()
        )
        for m in self.fc_output:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, pose_candidate: torch.Tensor):
        t = pose_candidate[:, :3]  # [B,3]
        B = t.shape[0]
        Pw = self.platform_points_local.unsqueeze(0).expand(B, -1, -1) + t.unsqueeze(1)  # [B,3,3]
        x  = Pw.reshape(B, -1)  # [B, 9]
        lengths = self.fc_output(x)  # [B,3]
        return Pw, lengths

# ================== Loss function: Geometric consistency + Travel constraint ==================
def custom_loss(pred_lengths: torch.Tensor, platform_points_global: torch.Tensor,
                base_points: torch.Tensor) -> torch.Tensor:
    base_points = base_points.to(platform_points_global.device)  # [3,3]
    leg_vec = platform_points_global - base_points.unsqueeze(0)  # [B,3,3]
    true_len = torch.norm(leg_vec, dim=2)  # [B,3]
    geo_loss = F.mse_loss(pred_lengths, true_len)

    stroke_lower = F.relu(l_min - pred_lengths)
    stroke_upper = F.relu(pred_lengths - l_max)
    stroke_penalty = (stroke_lower + stroke_upper).mean()

    return geo_loss + w_stroke * stroke_penalty

# ================== Data for training ==================
def synthesize_poses(n: int = 300000) -> np.ndarray:
    """
    generate (tx,ty,tz)[mm]：
      x,y ∈ [-150,150], z ∈ [-500,-100]
    """
    rng = np.random.default_rng(42)
    x = rng.uniform(-150.0, 150.0, size=(n, 1)).astype(np.float32)
    y = rng.uniform(-150.0, 150.0, size=(n, 1)).astype(np.float32)
    z = rng.uniform(-500.0, -100.0, size=(n, 1)).astype(np.float32)
    return np.hstack([x, y, z])

# ================== Training ==================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-data", type=int, default=500000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--model-out", type=str, default="best_model_delta.pth")
    ap.add_argument("--losses-out", type=str, default="losses.txt")
    args = ap.parse_args()

    x = synthesize_poses(args.num_data)            # (N,3)
    print(f"[info] synthesized poses: shape={x.shape}, "
          f"xyz_min={x.min(axis=0)}, xyz_max={x.max(axis=0)}")

    x_train, x_val = train_test_split(x, test_size=0.2, random_state=42)
    x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
    x_val   = torch.tensor(x_val,   dtype=torch.float32, device=device)

    model = DeltaPMP(PLAT_LOCAL, num_legs=3, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    batch_size = args.batch_size
    epochs = args.epochs
    num_batches = max(1, x_train.size(0) // batch_size)
    best_val, patience, wait = float("inf"), args.patience, 0
    losses, val_losses = [], []

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        perm = torch.randperm(x_train.size(0), device=device)
        for i in range(0, x_train.size(0), batch_size):
            idx = perm[i:i+batch_size]
            xb  = x_train[idx]  # [B,3]
            Pw, y_pred = model(xb)
            loss = custom_loss(y_pred, Pw, BASE_POINTS)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            Pw_val, y_val_pred = model(x_val)
            val_loss = custom_loss(y_val_pred, Pw_val, BASE_POINTS).item()

        scheduler.step()
        losses.append(epoch_loss / num_batches)
        val_losses.append(val_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: loss={losses[-1]:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_sd  = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    dt = time.time() - t0
    print(f"Total training time: {dt/3600.0:.2f} hours")
    torch.save(best_sd if 'best_sd' in locals() else model.state_dict(), args.model_out)
    print(f"[info] saved model to {args.model_out}")

    with open(args.losses_out, 'w') as f:
        for e, (tr, va) in enumerate(zip(losses, val_losses), start=1):
            f.write(f"Epoch {e}: Loss: {tr:.6f}, Val Loss: {va:.6f}\n")
    print(f"[info] saved losses to {args.losses_out}")

if __name__ == "__main__":
    main()




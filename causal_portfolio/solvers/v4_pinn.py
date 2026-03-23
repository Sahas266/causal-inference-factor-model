"""V4 PINN Solver: Physics-Informed Neural Network without HJB.

Learns a nonlinear driver→return mapping using an MLP with Jacobian
smoothness regularization. The smoothness penalty enforces manifold
regularity without solving the full HJB PDE.

Loss: L = L_return + λ_J * L_jacobian + λ_ridge * ||θ||²

This is the paper's best-performing variant (Combo-EKF-V4, m=3, Sharpe 1.266).

Reference: Section 4.4 of arXiv:2509.09585v2
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .base import CPCMSolver

logger = logging.getLogger("cpcm.solvers")


class _DriverReturnMLP(nn.Module):
    """MLP mapping m drivers → n_assets predicted returns."""

    def __init__(self, m_drivers: int, n_assets: int, hidden_dim: int, n_layers: int):
        super().__init__()
        layers = []
        in_dim = m_drivers
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_assets))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class V4PINNSolver(CPCMSolver):
    """PINN solver with Jacobian smoothness regularization (no HJB)."""

    def __init__(
        self,
        m_drivers: int,
        n_assets: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        device: str = "cpu",
    ):
        self.m_drivers = m_drivers
        self.n_assets = n_assets
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = torch.device(device)

        self.net = _DriverReturnMLP(
            m_drivers, n_assets, hidden_dim, n_layers
        ).to(self.device)
        self._fitted = False
        self._train_history: list[dict] = []

    def fit(
        self,
        drivers: np.ndarray,
        returns: np.ndarray,
        n_epochs: int = 500,
        lr: float = 1e-3,
        lambda_J: float = 0.01,
        lambda_ridge: float = 1e-4,
        val_fraction: float = 0.2,
        patience: int = 50,
        batch_size: Optional[int] = None,
    ) -> None:
        """Train the PINN on historical data.

        Args:
            drivers: (T, m) driver values.
            returns: (T, n_assets) asset returns.
            n_epochs: Maximum training epochs.
            lr: Learning rate.
            lambda_J: Jacobian smoothness penalty weight.
            lambda_ridge: Weight decay (L2 regularization).
            val_fraction: Fraction for temporal validation split.
            patience: Early stopping patience.
            batch_size: Mini-batch size (None = full batch).
        """
        # Drop NaN rows
        valid = ~(np.isnan(drivers).any(axis=1) | np.isnan(returns).any(axis=1))
        D = drivers[valid]
        R = returns[valid]
        T = len(D)

        # Temporal train/val split (no shuffling for time series)
        split = int(T * (1 - val_fraction))
        D_train = torch.tensor(D[:split], dtype=torch.float32, device=self.device)
        R_train = torch.tensor(R[:split], dtype=torch.float32, device=self.device)
        D_val = torch.tensor(D[split:], dtype=torch.float32, device=self.device)
        R_val = torch.tensor(R[split:], dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=lambda_ridge
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(n_epochs):
            self.net.train()

            # Forward pass
            if batch_size and batch_size < len(D_train):
                idx = torch.randperm(len(D_train))[:batch_size]
                D_batch = D_train[idx]
                R_batch = R_train[idx]
            else:
                D_batch = D_train
                R_batch = R_train

            pred = self.net(D_batch)
            loss_return = nn.functional.mse_loss(pred, R_batch)

            # Jacobian smoothness penalty: ||d²A/dF²||_F
            loss_jac = self._jacobian_smoothness(D_batch, lambda_J)

            loss = loss_return + loss_jac

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Validation
            self.net.eval()
            with torch.no_grad():
                val_pred = self.net(D_val)
                val_loss = nn.functional.mse_loss(val_pred, R_val).item()

            self._train_history.append({
                "epoch": epoch,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "lr": scheduler.get_last_lr()[0],
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Restore best model
        if best_state is not None:
            self.net.load_state_dict(best_state)

        self._fitted = True
        logger.info(
            f"V4 PINN fitted: {n_epochs} max epochs, "
            f"best val loss={best_val_loss:.6f}"
        )

    def predict(self, drivers: np.ndarray) -> np.ndarray:
        """Predict returns given driver state."""
        self._check_fitted()
        self.net.eval()
        if drivers.ndim == 1:
            drivers = drivers.reshape(1, -1)
        with torch.no_grad():
            x = torch.tensor(drivers, dtype=torch.float32, device=self.device)
            pred = self.net(x)
        return pred.cpu().numpy()

    def jacobian(self, drivers: np.ndarray) -> np.ndarray:
        """Compute dA/dF via autodiff. Returns (n_assets, m) matrix."""
        self._check_fitted()
        self.net.eval()
        x = torch.tensor(
            drivers.reshape(1, -1), dtype=torch.float32,
            device=self.device, requires_grad=True,
        )

        def net_fn(inp):
            return self.net(inp).squeeze(0)

        J = torch.autograd.functional.jacobian(net_fn, x)
        # J shape: (n_assets, 1, m) → squeeze to (n_assets, m)
        return J.squeeze(1).detach().cpu().numpy()

    def _jacobian_smoothness(
        self, D_batch: torch.Tensor, lambda_J: float,
    ) -> torch.Tensor:
        """Compute Jacobian smoothness penalty.

        Approximates ||d²A/dF²||_F via finite differences of the Jacobian
        at neighboring points in the batch for efficiency.
        """
        if lambda_J <= 0 or len(D_batch) < 3:
            return torch.tensor(0.0, device=self.device)

        # Sample a small subset for efficiency
        n_sample = min(16, len(D_batch))
        idx = torch.randperm(len(D_batch) - 1)[:n_sample]

        D_batch.requires_grad_(True)
        pred = self.net(D_batch)

        # Compute gradient of each output w.r.t. inputs for sampled points
        total_penalty = torch.tensor(0.0, device=self.device)
        for i in range(self.n_assets):
            grad = torch.autograd.grad(
                pred[:, i].sum(), D_batch,
                create_graph=True, retain_graph=True,
            )[0]  # (T, m) — gradient of asset i w.r.t. all drivers

            # Finite-difference approximation of second derivative
            grad_diff = grad[idx + 1] - grad[idx]
            total_penalty = total_penalty + (grad_diff ** 2).sum()

        return lambda_J * total_penalty / (n_sample * self.n_assets)

    @property
    def train_history(self) -> list[dict]:
        return self._train_history

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Solver not fitted. Call fit() first.")

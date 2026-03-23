"""Abstract base class for CPCM solvers.

All solver variants (V1 linear, V4 PINN, etc.) implement this interface.
The key abstraction is that every solver provides:
  - fit(): learn the driver→return mapping
  - predict(): expected returns given driver state
  - jacobian(): dA/dF sensitivity matrix (constant for linear, autodiff for NN)
"""

from abc import ABC, abstractmethod

import numpy as np


class CPCMSolver(ABC):
    """Base class for CPCM driver→return mapping solvers."""

    @abstractmethod
    def fit(self, drivers: np.ndarray, returns: np.ndarray) -> None:
        """Fit the solver on historical data.

        Args:
            drivers: (T, m) array of driver values.
            returns: (T, n_assets) array of asset returns.
        """

    @abstractmethod
    def predict(self, drivers: np.ndarray) -> np.ndarray:
        """Predict expected returns given driver state.

        Args:
            drivers: (T, m) or (m,) array of driver values.

        Returns:
            (T, n_assets) or (n_assets,) predicted returns.
        """

    @abstractmethod
    def jacobian(self, drivers: np.ndarray) -> np.ndarray:
        """Compute dA/dF — sensitivity of returns to driver changes.

        Args:
            drivers: (m,) current driver state.

        Returns:
            (n_assets, m) Jacobian matrix.
        """

    @property
    def is_fitted(self) -> bool:
        """Whether the solver has been fit."""
        return getattr(self, "_fitted", False)

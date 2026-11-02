#!/usr/bin/env python3
import math
from typing import List, Sequence

import torch
from torch import Tensor
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood


class LeaveGroupOutPseudoLikelihood(ExactMarginalLogLikelihood):
    """
    Leave-group-out pseudo log likelihood for an exact GP with Gaussian likelihood.

    Given a partition of the observations into non-overlapping groups S_1, ..., S_G,
    this objective maximizes

        sum_g log p(y_{S_g} | y_{-S_g}, X, theta)

    rather than the exact marginal log likelihood or leave-one-out pseudo likelihood.

    This is useful when one wants to hold out structured blocks of observations
    (e.g. all observations from a station, a time window, or a spatial cluster).

    Requirements:
      - ExactGP model
      - GaussianLikelihood
      - non-overlapping groups of indices

    Parameters
    ----------
    likelihood:
        A Gaussian likelihood.
    model:
        An ExactGP model.
    groups:
        A sequence of index collections. Each entry contains the indices of one held-out group.
        The groups should be non-overlapping, and typically form a partition of {0, ..., n-1}.

    Example
    -------
    >>> groups = [
    >>>     torch.tensor([0, 1, 2]),
    >>>     torch.tensor([3, 4]),
    >>>     torch.tensor([5, 6, 7, 8]),
    >>> ]
    >>> mll = LeaveGroupOutPseudoLikelihood(likelihood, model, groups)
    >>> output = model(train_x)
    >>> loss = -mll(output, train_y)
    >>> loss.backward()
    """

    def __init__(self, likelihood, model, groups: Sequence[Sequence[int]]):
        super().__init__(likelihood=likelihood, model=model)
        self.likelihood = likelihood
        self.model = model
        self.groups = groups

    def _validate_groups(self, n: int, device, dtype) -> List[Tensor]:
        """
        Convert input groups to LongTensor index arrays and validate that they are:
          - one-dimensional
          - non-empty
          - within bounds
          - non-overlapping
        """
        processed = []
        seen = set()

        for g, group in enumerate(self.groups):
            idx = torch.as_tensor(group, device=device, dtype=torch.long).flatten()

            if idx.numel() == 0:
                raise ValueError(f"Group {g} is empty.")

            if torch.any(idx < 0) or torch.any(idx >= n):
                raise ValueError(f"Group {g} contains indices outside [0, {n-1}].")

            unique_idx = torch.unique(idx)
            if unique_idx.numel() != idx.numel():
                raise ValueError(f"Group {g} contains repeated indices.")

            for i in idx.tolist():
                if i in seen:
                    raise ValueError(
                        f"Groups must be non-overlapping, but index {i} appears more than once."
                    )
                seen.add(i)

            processed.append(idx)

        return processed

    def forward(self, function_dist: MultivariateNormal, target: Tensor, *params) -> Tensor:
        r"""
        Compute the leave-group-out pseudo likelihood.

        Parameters
        ----------
        function_dist:
            Latent GP distribution p(f).
        target:
            Observation vector y.
        *params:
            Additional parameters passed into the likelihood.

        Returns
        -------
        Tensor
            Average leave-group-out pseudo log likelihood per observation.
        """
        output = self.likelihood(function_dist, *params)

        m = output.mean.reshape(*target.shape)
        K_lazy = output.lazy_covariance_matrix
        L = K_lazy.cholesky(upper=False)

        # For now, assume the final dimension indexes the data.
        n = target.size(-1)
        device = target.device
        dtype = target.dtype

        groups = self._validate_groups(n=n, device=device, dtype=dtype)

        # alpha = K^{-1}(y - m)
        residual = (target - m).unsqueeze(-1)  # (..., n, 1)
        alpha = L._cholesky_solve(residual, upper=False).squeeze(-1)  # (..., n)

        # Q = K^{-1}
        eye = torch.eye(n, device=device, dtype=dtype)
        Q = L._cholesky_solve(eye, upper=False)  # (..., n, n) or (n, n)

        total_log_prob = torch.zeros(target.shape[:-1], device=device, dtype=dtype)

        for idx in groups:
            # Extract block precision Q_SS and block alpha
            Q_ss = Q.index_select(-2, idx).index_select(-1, idx)   # (..., k, k)
            alpha_s = alpha.index_select(-1, idx)                  # (..., k)

            # Cholesky factorization of Q_SS
            # Q_SS should be SPD if K is SPD and idx is valid
            chol_Qss = torch.linalg.cholesky(Q_ss)

            # Quadratic term: alpha_s^T Q_SS^{-1} alpha_s
            # Compute x = Q_SS^{-1} alpha_s by triangular solves
            x = torch.cholesky_solve(alpha_s.unsqueeze(-1), chol_Qss).squeeze(-1)
            quad = (alpha_s * x).sum(dim=-1)

            # logdet(Q_SS) from Cholesky
            logdet_Qss = 2.0 * torch.log(torch.diagonal(chol_Qss, dim1=-2, dim2=-1)).sum(dim=-1)

            k = idx.numel()
            group_log_prob = -0.5 * (
                quad
                - logdet_Qss
                + k * math.log(2.0 * math.pi)
            )

            total_log_prob = total_log_prob + group_log_prob

        total_log_prob = self._add_other_terms(total_log_prob, params)

        # Normalize per observation, matching the style of LeaveOneOutPseudoLikelihood
        return total_log_prob / n
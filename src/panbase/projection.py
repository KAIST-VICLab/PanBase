# SPDX-License-Identifier: Apache-2.0
"""Observation operator D of the PanCollection reduced-resolution protocol and the projection Pi onto {x : D x = m}.

    D(x)_b = decimate_{r, offset}( k_b * pad_replicate(x_b) )

``k_b`` is the 41 x 41 MTF-matched kernel of band b (:mod:`panbase.mtf`), applied as a correlation on a replicate-padded
image; decimation keeps rows and columns ``offset, offset + r, ...`` (offset 2 for ratio 4).  With these conventions
``D(GT) = MS`` holds to round-off on PanCollection (WV3 / GF2 train, valid and test; QB / WV2 test).

The transpose D^T is never written by hand: D is linear, so its vector-Jacobian product is D^T exactly.

    Pi(u; m) = u + D^T (D D^T)^{-1} (m - D u)      orthogonal projection onto {x : D x = m}
    Q(z)     = z - D^T (D D^T)^{-1} D z            projector onto the null space of D (used by the fit)

``(D D^T)^{-1}`` is applied per (sample, band) by preconditioned conjugate gradients.  The preconditioner is the exact
inverse of the separable (Kronecker) approximation ``k_b ~ k_y k_x^T / sum(k_b)`` built from the kernel's marginals with
the same padding and decimation; the true D is used in every matrix-vector product.  The iteration stops when
``max ||res||^2 / ||res_0||^2 < tol`` over the whole batch (``tol = 1e-12``, at most 200 iterations), as in the paper.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from .mtf import mtf_kernels


@contextmanager
def _no_tf32(cuda: bool):
    """D is a physics operator: never run its float32 convolutions in TF32 on CUDA (about 1e-3 relative error).
    A no-op for CPU tensors."""
    if not cuda:
        yield
        return
    c = torch.backends.cudnn
    kw = dict(enabled=c.enabled, benchmark=c.benchmark, deterministic=c.deterministic, allow_tf32=False)
    if hasattr(c, "benchmark_limit"):
        kw["benchmark_limit"] = c.benchmark_limit
    with c.flags(**kw):
        yield


class MTFObservation:
    """D for one sensor: per-band MTF blur (replicate padding) followed by decimation by `ratio` at `offset`.

    Tensors are ``(N, C, H, W)`` with ``C == len(gnyq)``; ``H`` and ``W`` must be multiples of `ratio`.
    """

    def __init__(self, gnyq, ratio: int = 4, offset: int | None = None, kernel_size: int = 41,
                 dtype: torch.dtype = torch.float32, device="cpu"):
        if int(ratio) != ratio or ratio < 1:
            raise ValueError("ratio must be a positive integer")
        self.gnyq = [float(g) for g in gnyq]
        self.ratio = int(ratio)
        self.offset = self.ratio // 2 if offset is None else int(offset)
        self.kernel_size = int(kernel_size)
        self.pad = self.kernel_size // 2
        self.dtype, self.device = dtype, torch.device(device)
        k = mtf_kernels(self.gnyq, self.ratio, self.kernel_size)                  # (C, n, n) float64
        # the operator works with the kernel rounded to the compute dtype (float32 by default)
        self.weight = torch.as_tensor(np.ascontiguousarray(k), dtype=dtype, device=self.device)[:, None]
        self._kron_cache = {}

    def spec(self) -> dict:
        return dict(kind="pancollection-mtf", gnyq=self.gnyq, ratio=self.ratio, offset=self.offset,
                    kernel_size=self.kernel_size, padding="replicate")

    def to(self, device=None, dtype=None) -> "MTFObservation":
        """Same operator on another device / dtype (the kernel is re-derived from the float64 design)."""
        return MTFObservation(self.gnyq, self.ratio, self.offset, self.kernel_size, dtype or self.dtype,
                              device or self.device)

    # ------------------------------------------------------------------ D, D^T
    def D(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) -> (N, C, ceil((H - offset) / r), ceil((W - offset) / r))."""
        if x.shape[1] != self.weight.shape[0]:
            raise ValueError(f"expected {self.weight.shape[0]} bands, got {x.shape[1]}")
        with _no_tf32(x.is_cuda):
            xp = F.pad(x, (self.pad,) * 4, mode="replicate")
            w = self.weight.to(device=x.device, dtype=x.dtype)
            return F.conv2d(xp[..., self.offset:, self.offset:], w, stride=self.ratio, groups=x.shape[1])

    def Dt(self, v: torch.Tensor, hr_shape) -> torch.Tensor:
        """D^T v as the vector-Jacobian product of D (exact transpose)."""
        with torch.enable_grad(), _no_tf32(v.is_cuda):
            x = torch.zeros(tuple(hr_shape), device=v.device, dtype=v.dtype, requires_grad=True)
            (g,) = torch.autograd.grad(self.D(x), x, grad_outputs=v)
        return g

    # ------------------------------------------------------------------ Kronecker preconditioner
    def _decimation_matrix(self, k1: torch.Tensor, n: int) -> torch.Tensor:
        """(C, n_lr, n) float64 matrix of the 1-D operator: correlation with k1 (C, ks), replicate padding, decimation."""
        C, ks = k1.shape
        n_lr = -(-(n - self.offset) // self.ratio)
        rows = torch.arange(n_lr)[:, None]
        taps = torch.arange(ks)[None, :]
        cols = (self.offset + self.ratio * rows + taps - self.pad).clamp(0, n - 1)          # (n_lr, ks)
        M = torch.zeros(C, n_lr, n, dtype=torch.float64)
        idx = (rows * n + cols).reshape(-1).expand(C, -1)
        M.view(C, -1).scatter_add_(1, idx, k1[:, None, :].expand(C, n_lr, ks).reshape(C, -1))
        return M

    def kron(self, H: int, W: int, dtype=None) -> dict:
        """Eigen-decompositions of A_y = D_y D_y^T and A_x = D_x D_x^T (per band) of the separable approximation."""
        dtype = dtype or self.dtype
        key = (H, W, dtype, str(self.device))
        if key not in self._kron_cache:
            k = self.weight[:, 0].detach().to("cpu", torch.float64)                      # (C, ks, ks), compute-dtype values
            tot = k.sum((-2, -1)).sqrt()[:, None, None]
            Dy = self._decimation_matrix(k.sum(-1), H) / tot                             # marginal along y (sum over x)
            Dx = self._decimation_matrix(k.sum(-2), W) / tot
            ly, Uy = torch.linalg.eigh(Dy @ Dy.transpose(-1, -2))
            lx, Ux = torch.linalg.eigh(Dx @ Dx.transpose(-1, -2))
            cv = lambda t: t.to(device=self.device, dtype=dtype)
            self._kron_cache[key] = dict(Uy=cv(Uy), Ux=cv(Ux), ly=cv(ly), lx=cv(lx))
        return self._kron_cache[key]

    @staticmethod
    def _kron_inv(R: torch.Tensor, k: dict) -> torch.Tensor:
        T = torch.einsum("cyh,ncyx,cxw->nchw", k["Uy"], R, k["Ux"])
        T = T / (k["ly"][:, :, None] * k["lx"][:, None, :])[None]
        return torch.einsum("chy,ncyx,cwx->nchw", k["Uy"], T, k["Ux"])

    # ------------------------------------------------------------------ (D D^T)^{-1}, Pi, Q
    @torch.no_grad()
    def solve(self, r: torch.Tensor, hr_shape, tol: float = 1e-12, maxiter: int = 200) -> torch.Tensor:
        """(D D^T)^{-1} r per (sample, band) by PCG; `r` is on the LR grid, `hr_shape` the shape of D's input."""
        H, W = hr_shape[-2:]
        k = self.kron(H, W, r.dtype)
        k = {a: b.to(r.device) for a, b in k.items()}
        A = lambda p: self.D(self.Dt(p, hr_shape))
        z = torch.zeros_like(r)
        res = r.clone()
        zz = self._kron_inv(res, k)
        p = zz.clone()
        rz = (res * zz).sum((-2, -1), keepdim=True)
        n0 = (res * res).sum((-2, -1), keepdim=True).clamp_min(1e-30)
        for _ in range(maxiter):
            Ap = A(p)
            a = rz / (p * Ap).sum((-2, -1), keepdim=True).clamp_min(1e-30)
            z = z + a * p
            res = res - a * Ap
            rn = (res * res).sum((-2, -1), keepdim=True)
            if float((rn / n0).max()) < tol:
                break
            zz = self._kron_inv(res, k)
            rz_new = (res * zz).sum((-2, -1), keepdim=True)
            p = zz + (rz_new / rz.clamp_min(1e-30)) * p
            rz = rz_new
        return z

    @torch.no_grad()
    def project(self, u: torch.Tensor, m: torch.Tensor, tol: float = 1e-12, maxiter: int = 200) -> torch.Tensor:
        """Pi(u; m): the point of {x : D x = m} closest to u (per sample and band)."""
        return u + self.Dt(self.solve(m - self.D(u), u.shape, tol, maxiter), u.shape)

    @torch.no_grad()
    def null(self, z: torch.Tensor, tol: float = 1e-12, maxiter: int = 200) -> torch.Tensor:
        """Q(z): the component of z in the null space of D."""
        return z - self.Dt(self.solve(self.D(z), z.shape, tol, maxiter), z.shape)

    @torch.no_grad()
    def consistency(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Per-sample ||D x - m|| / ||m||."""
        d = (self.D(x) - m).flatten(1).norm(dim=1)
        return d / m.flatten(1).norm(dim=1).clamp_min(1e-30)


def pancollection_operator(dataset: str, dtype=torch.float32, device="cpu") -> MTFObservation:
    """D of a PanCollection dataset ('wv3', 'qb', 'gf2', 'wv2'): ratio 4, offset 2, published Nyquist gains."""
    from .mtf import GNYQ
    return MTFObservation(GNYQ[dataset], 4, 2, 41, dtype, device)


__all__ = ["MTFObservation", "pancollection_operator"]

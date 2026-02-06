from torch.distributions.normal import Normal
import torch
import torch.nn as nn
import math

# ----------------------------
# Helpers (vectorized, batch-friendly)
# ----------------------------

def _safe_renorm(w: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Safely renormalize along `dim` (keeps all-zeros rows as zeros)."""
    s = w.sum(dim=dim, keepdim=True)
    return torch.where(s > 0, w / s.clamp_min(eps), w)

def pack_to_kmax(pi, mu, sigma, eps=1e-12, min_components=1, max_components=None):
    """
    Zero weights <= eps, then sort components by descending weight and keep only:

      Kkeep = min( Kmax_nonzero_global , max_components )   (if max_components is set)

    where Kmax_nonzero_global is the maximum number of non-zero weights across all mixtures
    in the tensor.

    Inputs:
      pi, mu, sigma: (..., K)

    Returns:
      pi_p, mu_p, sigma_p: (..., Kkeep)
    """
    # Threshold
    pi = pi * (pi > eps)

    # Sort by weight so zeros go to the end
    order = torch.argsort(pi, dim=-1, descending=True)
    pi_s = torch.gather(pi, dim=-1, index=order)
    mu_s = torch.gather(mu, dim=-1, index=order)
    sigma_s = torch.gather(sigma, dim=-1, index=order)

    # Count non-zeros per mixture
    nnz = (pi_s > 0).sum(dim=-1)  # (...,)

    # Global Kmax across all mixtures in this tensor
    Kmax = int(nnz.max().item()) if nnz.numel() > 0 else 0
    Kmax = max(Kmax, min_components)

    # Optional: cap the number of components
    if max_components is not None:
        Kmax = min(Kmax, int(max_components))

    # Keep only first Kmax components
    pi_s = pi_s[..., :Kmax]
    mu_s = mu_s[..., :Kmax]
    sigma_s = sigma_s[..., :Kmax]

    # Renormalize safely
    s = pi_s.sum(dim=-1, keepdim=True)
    pi_s = torch.where(s > 0, pi_s / s.clamp_min(eps), pi_s)

    return pi_s, mu_s, sigma_s


def gm_add_nd(pi1, mu1, sg1, pi2, mu2, sg2, eps=1e-12, pack=True, max_components=None):
    w = (pi1.unsqueeze(-1) * pi2.unsqueeze(-2))
    mu = (mu1.unsqueeze(-1) + mu2.unsqueeze(-2))
    var = (sg1.unsqueeze(-1) ** 2 + sg2.unsqueeze(-2) ** 2)
    sg = torch.sqrt(var.clamp_min(eps))

    K1 = pi1.shape[-1]
    K2 = pi2.shape[-1]
    out_shape = w.shape[:-2] + (K1 * K2,)
    w = w.reshape(out_shape)
    mu = mu.reshape(out_shape)
    sg = sg.reshape(out_shape)

    if pack:
        w, mu, sg = pack_to_kmax(w, mu, sg, eps=eps, min_components=1, max_components=max_components)
    else:
        w = w * (w > eps)
        w = _safe_renorm(w, dim=-1, eps=eps)

    return w, mu, sg


def truncate_0_vectorized(pi, mu, sigma, eps: float = 1e-12, pack: bool = True, max_components=None):
    """
    Vectorized + batch-friendly:
      inputs:  (..., K)
      outputs: (..., K+1)  (last component = delta at 0)
    """
    std = Normal(
        torch.tensor(0.0, device=pi.device, dtype=pi.dtype),
        torch.tensor(1.0, device=pi.device, dtype=pi.dtype),
    )

    sigma_safe = sigma.clamp_min(eps)
    alpha = -mu / sigma_safe

    # P(X>0) = 1 - Phi(alpha)
    prob_mass = (1.0 - std.cdf(alpha)).clamp_min(eps)

    # phi(alpha)
    phi = std.log_prob(alpha).exp()
    ratio = phi / prob_mass

    mu_trunc = mu + sigma_safe * ratio
    var_trunc = sigma_safe**2 * (1.0 + alpha * ratio - ratio**2)
    sigma_trunc = torch.sqrt(var_trunc.clamp_min(eps))

    pi_trunc = pi * prob_mass
    prob_zero = (pi * (1.0 - prob_mass)).sum(dim=-1, keepdim=True)

    pi_new = torch.cat([pi_trunc, prob_zero], dim=-1)
    zeros = torch.zeros_like(prob_zero)
    mu_new = torch.cat([mu_trunc, zeros], dim=-1)
    sigma_new = torch.cat([sigma_trunc, zeros], dim=-1)

    if pack:
        pi_new, mu_new, sigma_new = pack_to_kmax(
            pi_new, mu_new, sigma_new, eps=eps, min_components=1, max_components=max_components
        )
    else:
        pi_new = pi_new * (pi_new > eps)
        pi_new = _safe_renorm(pi_new, dim=-1, eps=eps)

    return pi_new, mu_new, sigma_new


def gm_product_layer(pi_w, mu_w, sg_w, pi_x, mu_x, sg_x,
                     eps: float = 1e-12, normalize_pi: bool = True, pack: bool = True, max_components=None):
    """
    Compute mixtures for (w_{j,i} * x_i) for all neurons j and inputs i.

    Weights:
      pi_w, mu_w, sg_w: (J, I, Kw)

    Inputs:
      pi_x, mu_x, sg_x: (B, I, Kx)  or (I, Kx) (broadcasts to B=1)

    Returns:
      pi_y, mu_y, sg_y: (B, J, I, Kw*Kx)   (or packed to <= max_components if enabled)
    """
    J, I, Kw = pi_w.shape

    # Promote x to batched: (B, I, Kx)
    if pi_x.dim() == 2:
        pi_x = pi_x.unsqueeze(0)
        mu_x = mu_x.unsqueeze(0)
        sg_x = sg_x.unsqueeze(0)

    B, I2, Kx = pi_x.shape
    assert I == I2, f"I mismatch: weights have {I}, inputs have {I2}"

    # Broadcast:
    # weights: (1, J, I, Kw, 1)
    # inputs:  (B, 1, I, 1, Kx)
    pi = pi_w.unsqueeze(0).unsqueeze(-1) * pi_x.unsqueeze(1).unsqueeze(-2)
    mu = mu_w.unsqueeze(0).unsqueeze(-1) * mu_x.unsqueeze(1).unsqueeze(-2)

    sw2 = (sg_w.unsqueeze(0).unsqueeze(-1) ** 2)
    sx2 = (sg_x.unsqueeze(1).unsqueeze(-2) ** 2)
    mw2 = (mu_w.unsqueeze(0).unsqueeze(-1) ** 2)
    mx2 = (mu_x.unsqueeze(1).unsqueeze(-2) ** 2)

    var = sw2 * sx2 + sw2 * mx2 + sx2 * mw2
    sg = torch.sqrt(var.clamp_min(eps))

    # Flatten (Kw, Kx) -> Kprod
    Kprod = Kw * Kx
    pi = pi.reshape(B, J, I, Kprod)
    mu = mu.reshape(B, J, I, Kprod)
    sg = sg.reshape(B, J, I, Kprod)

    if normalize_pi:
        if pack:
            pi, mu, sg = pack_to_kmax(pi, mu, sg, eps=eps, min_components=1, max_components=max_components)
        else:
            pi = pi * (pi > eps)
            pi = _safe_renorm(pi, dim=-1, eps=eps)

    return pi, mu, sg


def sum_over_inputs_tree_layer(pi, mu, sg, eps: float = 1e-12, max_components=None):
    """
    Tree-reduce sum over I dimension:
      pi, mu, sg: (B, J, I, K)

    Returns:
      pi_z, mu_z, sg_z: (B, J, K_out)
    """
    B, J, I, K = pi.shape
    level = [(pi[:, :, i, :], mu[:, :, i, :], sg[:, :, i, :]) for i in range(I)]

    while len(level) > 1:
        new = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                pi1, mu1, sg1 = level[i]
                pi2, mu2, sg2 = level[i + 1]
                new.append(gm_add_nd(pi1, mu1, sg1, pi2, mu2, sg2, eps=eps, pack=True, max_components=max_components))
            else:
                new.append(level[i])
        level = new

    return level[0]


def layer_forward_gmm(layer, pi_x, mu_x, sg_x,
                      eps: float = 1e-12, apply_relu: bool = True, max_components=None):
    """
    Forward for one BayesLinearGMM layer:
      z = W x + b   (mixture propagation)
      a = relu(z)   (optional, via truncate_0_vectorized)

    Returns:
      (pi, mu, sg): (B, out, K_out)  (or (out, K_out) if unbatched input)
    """
    unbatched = (pi_x.dim() == 2)

    # 1) products (B, out, in, Kw*Kx)
    pi_y, mu_y, sg_y = gm_product_layer(
        layer.pi_w, layer.mu_w, layer.sigma_w,
        pi_x, mu_x, sg_x,
        eps=eps, normalize_pi=True, pack=True, max_components=max_components
    )

    # 2) sum over inputs
    pi_z, mu_z, sg_z = sum_over_inputs_tree_layer(pi_y, mu_y, sg_y, eps=eps, max_components=max_components)

    # 3) add bias
    if layer.bias:
        pi_b = layer.pi_b.unsqueeze(0)
        mu_b = layer.mu_b.unsqueeze(0)
        sg_b = layer.sigma_b.unsqueeze(0)
        pi_z, mu_z, sg_z = gm_add_nd(
            pi_z, mu_z, sg_z, pi_b, mu_b, sg_b,
            eps=eps, pack=True, max_components=max_components
        )

    # 4) activation
    if apply_relu:
        pi_a, mu_a, sg_a = truncate_0_vectorized(pi_z, mu_z, sg_z, eps=eps, pack=True, max_components=max_components)
    else:
        pi_a, mu_a, sg_a = pi_z, mu_z, sg_z

    if unbatched:
        return pi_a.squeeze(0), mu_a.squeeze(0), sg_a.squeeze(0)
    return pi_a, mu_a, sg_a

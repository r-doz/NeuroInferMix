import torch
import torch.nn as nn
import math
from representation.operations_advanced_pruning import layer_forward_gmm

class BayesLinearGMM(nn.Module):
    """
    Fully-connected layer where each scalar weight/bias is a K-component univariate Gaussian mixture.
    Shapes:
      mu_w, sigma_w, pi_w: (out, in, K)
      mu_b, sigma_b, pi_b: (out, K)
    """
    def __init__(self, in_features: int, out_features: int, K: int, bias: bool = True, eps: float = 1e-12):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.bias = bias
        self.eps = eps

        self.pi_w = nn.Parameter(torch.empty(out_features, in_features, K))
        self.mu_w = nn.Parameter(torch.empty(out_features, in_features, K))
        self.sigma_w = nn.Parameter(torch.empty(out_features, in_features, K))

        if bias:
            self.pi_b = nn.Parameter(torch.empty(out_features, K))
            self.mu_b = nn.Parameter(torch.empty(out_features, K))
            self.sigma_b = nn.Parameter(torch.empty(out_features, K))
        else:
            self.register_parameter("pi_b", None)
            self.register_parameter("mu_b", None)
            self.register_parameter("sigma_b", None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        fan_in = self.in_features
        bound = 1.0 / math.sqrt(fan_in)

        self.mu_w.uniform_(-bound, bound)
        self.mu_w.add_(0.01 * torch.randn_like(self.mu_w))
        self.pi_w.fill_(1.0 / self.K)
        self.sigma_w.fill_(0.05 * bound)

        if self.bias:
            self.mu_b.uniform_(-bound, bound)
            self.mu_b.add_(0.01 * torch.randn_like(self.mu_b))
            self.pi_b.fill_(1.0 / self.K)
            self.sigma_b.fill_(0.05 * bound)

    @torch.no_grad()
    def init_from_deterministic(self, W: torch.Tensor, b: torch.Tensor | None = None,
                                sigma0: float = 1e-3, main_comp: int = 0):
        assert W.shape == (self.out_features, self.in_features)

        self.mu_w.zero_()
        self.mu_w[..., :] = 0.01 * torch.randn_like(self.mu_w)
        self.mu_w[..., main_comp] = W

        self.sigma_w.fill_(sigma0)

        self.pi_w.fill_(self.eps)
        self.pi_w[..., main_comp] = 1.0
        self.pi_w.div_(self.pi_w.sum(dim=-1, keepdim=True))

        if self.bias:
            if b is None:
                b = torch.zeros(self.out_features, device=W.device, dtype=W.dtype)
            assert b.shape == (self.out_features,)

            self.mu_b.zero_()
            self.mu_b[:, :] = 0.01 * torch.randn_like(self.mu_b)
            self.mu_b[:, main_comp] = b

            self.sigma_b.fill_(sigma0)

            self.pi_b.fill_(self.eps)
            self.pi_b[:, main_comp] = 1.0
            self.pi_b.div_(self.pi_b.sum(dim=-1, keepdim=True))


class BNN_GMM(nn.Module):
    def __init__(self, layer_sizes, K: int, bias: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            BayesLinearGMM(layer_sizes[i], layer_sizes[i+1], K=K, bias=bias)
            for i in range(len(layer_sizes) - 1)
        ])

    @torch.no_grad()
    def init_from_deterministic_mlp(self, det, sigma0=1e-3, main_comp=0):
        if isinstance(det, (list, tuple)):
            det_linears = list(det)
        else:
            det_linears = [m for m in det.modules() if isinstance(m, nn.Linear)]

        if len(det_linears) != len(self.layers):
            raise ValueError(
                f"Mismatch: deterministic has {len(det_linears)} Linear layers, "
                f"BNN has {len(self.layers)} Bayesian layers.\n"
                f"Det linears: {[ (l.in_features, l.out_features) for l in det_linears ]}\n"
                f"BNN layers:  {[ (l.in_features, l.out_features) for l in self.layers ]}"
            )

        for bayes_layer, det_layer in zip(self.layers, det_linears):
            bayes_layer.init_from_deterministic(
                W=det_layer.weight.data,
                b=det_layer.bias.data,
                sigma0=sigma0,
                main_comp=main_comp
            )

    def forward(self, pi_x, mu_x, sg_x, eps: float | None = None, last_relu: bool = False, max_components=None):
        """
        Full mixture forward through all Bayesian layers.

        Inputs:
          pi_x, mu_x, sg_x: (B, d_in, Kx) or (d_in, Kx)

        Hidden layers: apply ReLU clamp (truncate_0_vectorized)
        Last layer: no ReLU by default (set last_relu=True if you want it)

        Returns:
          (pi, mu, sg): (B, d_out, K_out) (or (d_out, K_out) if input unbatched)
        """
        if eps is None:
            eps = self.layers[0].eps if len(self.layers) > 0 else 1e-12

        pi, mu, sg = pi_x, mu_x, sg_x

        L = len(self.layers)
        for ell, layer in enumerate(self.layers):
            is_last = (ell == L - 1)
            apply_relu = (not is_last) or last_relu

            pi, mu, sg = layer_forward_gmm(
                layer, pi, mu, sg,
                eps=eps,
                apply_relu=apply_relu,
                max_components = max_components
            )

        return pi, mu, sg

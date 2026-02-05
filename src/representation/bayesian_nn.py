import torch
import torch.nn as nn
import math

class BayesLinearGMM(nn.Module):
    """
    Fully-connected layer where each scalar weight/bias is a K-component univariate Gaussian mixture.
    Parameters are stored explicitly as pi/mu/sigma (no softmax/softplus).
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

        # Mixture params for weights
        self.pi_w = nn.Parameter(torch.empty(out_features, in_features, K))
        self.mu_w = nn.Parameter(torch.empty(out_features, in_features, K))
        self.sigma_w = nn.Parameter(torch.empty(out_features, in_features, K))

        # Mixture params for biases
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
        """
        Default init (reasonable, stable):
          - mu: Xavier/He-like scale for component means
          - pi: uniform over components
          - sigma: small positive noise scale
        If you already have “ok params”, you can overwrite after construction.
        """
        # Means: like nn.Linear init but replicated across K with small jitter
        fan_in = self.in_features
        bound = 1.0 / math.sqrt(fan_in)

        self.mu_w.uniform_(-bound, bound)
        self.mu_w.add_(0.01 * torch.randn_like(self.mu_w))  # tiny spread across components

        # Weights: start with uniform mixture
        self.pi_w.fill_(1.0 / self.K)

        # Small positive sigma (scaled to fan_in)
        self.sigma_w.fill_(0.05 * bound)

        if self.bias:
            self.mu_b.uniform_(-bound, bound)
            self.mu_b.add_(0.01 * torch.randn_like(self.mu_b))
            self.pi_b.fill_(1.0 / self.K)
            self.sigma_b.fill_(0.05 * bound)

    @torch.no_grad()
    def init_from_deterministic(self, W: torch.Tensor, b: torch.Tensor | None = None,
                                sigma0: float = 1e-3, main_comp: int = 0):
        """
        Useful if you have a trained deterministic net and want to “lift” it to mixtures:
          - Put W (and b) into one component mean
          - Give it high weight
          - Other components get near-zero weight and small random means
        """
        assert W.shape == (self.out_features, self.in_features)

        # Set all means near 0, then put deterministic weights into one component
        self.mu_w.zero_()
        self.mu_w[..., :] = 0.01 * torch.randn_like(self.mu_w)
        self.mu_w[..., main_comp] = W

        self.sigma_w.fill_(sigma0)

        self.pi_w.fill_(self.eps)  # tiny
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
        """
        layer_sizes: e.g. [d_in, h1, h2, d_out]
        """
        super().__init__()
        self.layers = nn.ModuleList([
            BayesLinearGMM(layer_sizes[i], layer_sizes[i+1], K=K, bias=bias)
            for i in range(len(layer_sizes) - 1)
        ])

    @torch.no_grad()
    def init_from_deterministic_mlp(self, det, sigma0=1e-3, main_comp=0):
        # det can be an nn.Module OR a list/tuple of nn.Linear
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
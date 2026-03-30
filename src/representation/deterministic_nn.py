import torch
import torch.nn as nn
import math
from representation.operations_advanced_pruning import det_layer_forward_gmm

class DeterministicLinearGMM(nn.Module):
    """
    Fully-connected layer where weights are scalars, but the input and the output are a gaussian mixture of max K components.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.W = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.b = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("b", None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """Xavier uniform initialization."""
        fan_in = self.in_features
        bound = 1.0 / math.sqrt(fan_in)

        self.W.uniform_(-bound, bound)
        if self.bias:
            self.b.uniform_(-bound, bound)

        
    @torch.no_grad()
    def init_from_deterministic(self, W: torch.Tensor, b: torch.Tensor | None = None):
        """
        Initialize the layer from a deterministic weight matrix and optional bias vector.

        Args:
            W (torch.Tensor): Deterministic weight matrix of shape (out_features, in_features).
            b (torch.Tensor | None): Optional deterministic bias vector of shape (out_features,).

        """
        assert W.shape == (self.out_features, self.in_features)

        self.W = W

        if self.bias:
            if b is None:
                b = torch.zeros(self.out_features, device=W.device, dtype=W.dtype)
            assert b.shape == (self.out_features,)

            self.b = b


class DNN_GMM(nn.Module):
    def __init__(self, layer_sizes, bias: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            DeterministicLinearGMM(layer_sizes[i], layer_sizes[i+1], bias=bias)
            for i in range(len(layer_sizes) - 1)
        ])

    @torch.no_grad()
    def init_from_deterministic_mlp(self, det):
        if isinstance(det, (list, tuple)):
            det_linears = list(det)
        else:
            det_linears = [m for m in det.modules() if isinstance(m, nn.Linear)]

        if len(det_linears) != len(self.layers):
            raise ValueError(
                f"Mismatch: deterministic has {len(det_linears)} Linear layers, "
                f"DNN has {len(self.layers)} Deterministic layers.\n"
                f"Det linears: {[ (l.in_features, l.out_features) for l in det_linears ]}\n"
                f"DNN layers:  {[ (l.in_features, l.out_features) for l in self.layers ]}"
            )

        for new_layer, det_layer in zip(self.layers, det_linears):
            new_layer.init_from_deterministic(
                W=det_layer.weight,
                b=det_layer.bias
            )


    def forward(self, pi_x, mu_x, sg_x, last_relu: bool = False, max_components=None):
        """
        Full mixture forward through all Bayesian layers.

        Inputs:
          pi_x, mu_x, sg_x: (B, d_in, Kx) or (d_in, Kx)

        Hidden layers: apply ReLU clamp (truncate_0_vectorized)
        Last layer: no ReLU by default (set last_relu=True if you want it)

        Returns:
          (pi, mu, sg): (B, d_out, K_out) (or (d_out, K_out) if input unbatched)
        """
        
        pi, mu, sg = pi_x, mu_x, sg_x

        L = len(self.layers)
        for ell, layer in enumerate(self.layers):
            is_last = (ell == L - 1)
            apply_relu = (not is_last) or last_relu

            pi, mu, sg = det_layer_forward_gmm(
                layer, pi, mu, sg,
                apply_relu=apply_relu,
                max_components = max_components
            )

        return pi, mu, sg

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

# ---------- metrics helpers ----------

def classification_accuracy_from_gmm_output(pi, mu, y_onehot):
    """
    pi, mu: (B, 10, K)
    y_onehot: (B, 10)
    """
    # Mixture mean per class => (B,10)
    y_hat = (pi * mu).sum(dim=-1)

    pred = torch.argmax(y_hat, dim=1)          # (B,)
    true = torch.argmax(y_onehot, dim=1)       # (B,)
    return (pred == true).float().mean().item()

def mixture_mean(pi, mu):
    # pi, mu: (B, D, K) -> (B, D)
    return (pi * mu).sum(dim=-1)

def mixture_variance(pi, mu, sg, eps=1e-12):
    mean = (pi * mu).sum(dim=-1)
    second = (pi * (sg**2 + mu**2)).sum(dim=-1)
    return second - mean**2

def regression_metrics_from_output(pi, mu, sg, y, eps=1e-12):
    """
    pi,mu,sg: (B,D,K)
    y:        (B,D)
    """
    y_hat = mixture_mean(pi, mu)
    mse = ((y_hat - y) ** 2).mean()
    rmse = torch.sqrt(mse)
    mae = (y_hat - y).abs().mean()

    var = mixture_variance(pi, mu, sg, eps=eps)
    std = torch.sqrt(var.clamp_min(eps))
    cov_1sigma = (((y >= y_hat - std) & (y <= y_hat + std)).float().mean())

    return {
        "MSE": mse.item(),
        "RMSE": rmse.item(),
        "MAE": mae.item(),
        "Cov@1σ": cov_1sigma.item(),
    }

def pad_gmm_to_K(pi, mu, sg, K_target, eps=1e-12):
    B, D, K = pi.shape
    if K == K_target:
        return pi, mu, sg
    if K > K_target:
        # better not to happen if K_target = global max
        pi = pi[:, :, :K_target]
        mu = mu[:, :, :K_target]
        sg = sg[:, :, :K_target]
        s = pi.sum(dim=-1, keepdim=True)
        pi = torch.where(s > eps, pi / s, pi)
        return pi, mu, sg

    pad = K_target - K
    pi_pad = torch.zeros(B, D, pad, device=pi.device, dtype=pi.dtype)
    mu_pad = torch.zeros(B, D, pad, device=mu.device, dtype=mu.dtype)
    sg_pad = torch.ones(B, D, pad, device=sg.device, dtype=sg.dtype)
    return (
        torch.cat([pi, pi_pad], dim=-1),
        torch.cat([mu, mu_pad], dim=-1),
        torch.cat([sg, sg_pad], dim=-1),
    )


def split_dataset(
    X,
    y,
    test_ratio=0.2,
    batch_size=64,
    seed=0,
):
    """
    Splits X,y into train/test loaders.
    Returns: (dl_train, dl_test)
    """
    ds = TensorDataset(X, y)
    n_total = len(ds)
    n_test = int(round(test_ratio * n_total))
    n_train = n_total - n_test

    g = torch.Generator().manual_seed(seed)
    ds_train, ds_test = random_split(ds, [n_train, n_test], generator=g)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    return dl_train, dl_test


def train_bnn(
    bnn,
    dl_train,
    loss_function,
    epochs=20,
    lr=1e-3,
    input_sigma=1e-2,
    eps=1e-12,
    last_relu=False,
    device=None,
    max_components=None,
):
    """
    Trains the BNN on a provided train loader.
    Returns: list of train losses per epoch.
    """
    if device is None:
        device = next(bnn.parameters()).device

    opt = torch.optim.Adam(bnn.parameters(), lr=lr)
    losses = []

    bnn.train()
    for ep in range(epochs):
        running = 0.0
        n_batches = 0

        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)

            # Deterministic inputs -> GM inputs (B, d_in, 1)
            pi_x = torch.ones(xb.shape[0], xb.shape[1], 1, device=device, dtype=xb.dtype)
            mu_x = xb.unsqueeze(-1)
            sg_x = input_sigma * torch.ones_like(mu_x)

            pi_out, mu_out, sg_out = bnn(
                pi_x, mu_x, sg_x, eps=eps, last_relu=last_relu, max_components=max_components
            )

            loss = loss_function(pi_out, mu_out, sg_out, yb, eps=eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            project_bnn_params_(bnn, eps=eps)

            running += loss.item()
            n_batches += 1

        print(f"epoch {ep+1}/{epochs} | train NLL {running / max(1,n_batches):.6f}")
        losses.append(running / max(1, n_batches))

    return losses


def evaluate_bnn(
    bnn,
    dl_test,
    loss_function,
    input_sigma=1e-2,
    eps=1e-12,
    last_relu=False,
    device=None,
    max_components=None,
):
    """
    Evaluates the BNN on a provided test loader.
    Returns: dict with test metrics (including NLL)
    """
    if device is None:
        device = next(bnn.parameters()).device

    bnn.eval()
    all_pi, all_mu, all_sg, all_y = [], [], [], []
    K_max = 0

    with torch.no_grad():
        for xb, yb in dl_test:
            xb = xb.to(device)
            yb = yb.to(device)

            pi_x = torch.ones(xb.shape[0], xb.shape[1], 1, device=device, dtype=xb.dtype)
            mu_x = xb.unsqueeze(-1)
            sg_x = input_sigma * torch.ones_like(mu_x)

            pi_out, mu_out, sg_out = bnn(
                pi_x, mu_x, sg_x, eps=eps, last_relu=last_relu, max_components=max_components
            )

            K_max = max(K_max, pi_out.shape[-1])

            all_pi.append(pi_out)
            all_mu.append(mu_out)
            all_sg.append(sg_out)
            all_y.append(yb)

    # pad to the largest K seen in the whole test set
    all_pi2, all_mu2, all_sg2 = [], [], []
    for pi_b, mu_b, sg_b in zip(all_pi, all_mu, all_sg):
        pi_b, mu_b, sg_b = pad_gmm_to_K(pi_b, mu_b, sg_b, K_max, eps=eps)
        all_pi2.append(pi_b)
        all_mu2.append(mu_b)
        all_sg2.append(sg_b)

    pi_out = torch.cat(all_pi2, dim=0)
    mu_out = torch.cat(all_mu2, dim=0)
    sg_out = torch.cat(all_sg2, dim=0)
    y_test = torch.cat(all_y, dim=0)

    test_nll = loss_function(pi_out, mu_out, sg_out, y_test, eps=eps).item()
    test_metrics = regression_metrics_from_output(pi_out, mu_out, sg_out, y_test, eps=eps)
    test_metrics["NLL"] = test_nll

    acc = classification_accuracy_from_gmm_output(pi_out, mu_out, y_test)
    test_metrics["Acc"] = acc
    print("TEST Acc:", acc)

    print("\nTEST metrics:", test_metrics)
    return test_metrics



@torch.no_grad()
def project_bnn_params_(bnn, eps=1e-12):
    for layer in bnn.layers:
        # keep sigmas positive
        layer.sigma_w.clamp_(min=eps)
        if layer.bias:
            layer.sigma_b.clamp_(min=eps)

        # keep pis valid probabilities
        layer.pi_w.clamp_(min=0.0)
        layer.pi_w.div_(layer.pi_w.sum(dim=-1, keepdim=True).clamp_min(eps))

        if layer.bias:
            layer.pi_b.clamp_(min=0.0)
            layer.pi_b.div_(layer.pi_b.sum(dim=-1, keepdim=True).clamp_min(eps))

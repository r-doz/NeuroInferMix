import torch
from torchvision import datasets, transforms

def load_mnist_onehot_tensors(
    root="./data",
    train=True,
    n_max=None,
    normalize=True,
    device="cpu",
    dtype=torch.float32,
):
    """
    Returns:
      X: (N, 784) float
      y: (N, 10)  one-hot float
    """
    if normalize:
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # standard MNIST stats
        ])
    else:
        tfm = transforms.ToTensor()

    ds = datasets.MNIST(root=root, train=train, download=True, transform=tfm)

    if n_max is None:
        n_max = len(ds)
    n_max = min(n_max, len(ds))

    X_list = []
    y_list = []

    for i in range(n_max):
        x_i, y_i = ds[i]              # x_i: (1,28,28), y_i: int in [0..9]
        X_list.append(x_i.view(-1))   # -> (784,)
        y_list.append(int(y_i))

    X = torch.stack(X_list, dim=0).to(device=device, dtype=dtype)  # (N,784)

    y_idx = torch.tensor(y_list, device=device, dtype=torch.long)  # (N,)
    y = torch.zeros((n_max, 10), device=device, dtype=dtype)
    y.scatter_(1, y_idx.unsqueeze(1), 1.0)                          # (N,10) one-hot

    return X, y
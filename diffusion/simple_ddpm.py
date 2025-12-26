import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm


# -----------------------------
# Tiny denoiser
# -----------------------------
class SimpleDenoiser(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.out = nn.Conv2d(hidden, 1, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, hidden)
        self.gn2 = nn.GroupNorm(8, hidden)
        self.gn3 = nn.GroupNorm(8, hidden)

    def forward(self, x, t):
        h = F.silu(self.gn1(self.conv1(x)))
        h = F.silu(self.gn2(self.conv2(h)))
        h = F.silu(self.gn3(self.conv3(h)))
        return self.out(h)


# -----------------------------
# Diffusion helpers
# -----------------------------
def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

def extract(a, t, x_shape):
    # a: [T], t: [B]
    out = a.gather(0, t).float()
    return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))

def q_sample(x0, t, noise, sqrt_ab, sqrt_1mab):
    return extract(sqrt_ab, t, x0.shape) * x0 + extract(sqrt_1mab, t, x0.shape) * noise


@torch.no_grad()
def p_sample(model, x, t, betas, alphas, ab, device):
    # Predict noise
    eps = model(x, t)

    beta_t = extract(betas, t, x.shape)
    alpha_t = extract(alphas, t, x.shape)
    ab_t = extract(ab, t, x.shape)

    # mean
    mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - ab_t)) * eps)

    # noise
    noise = torch.randn_like(x)
    nonzero = (t != 0).float().view(x.shape[0], *((1,) * (len(x.shape) - 1)))
    return mean + nonzero * torch.sqrt(beta_t) * noise


def get_loader(batch_size=128):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [0,1] -> [-1,1]
    ])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def train(epochs=5, T=200, lr=2e-4, ckpt="diffusion_ddpm_mnist.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleDenoiser().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = get_loader()

    betas = linear_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    ab = torch.cumprod(alphas, dim=0)
    sqrt_ab = torch.sqrt(ab)
    sqrt_1mab = torch.sqrt(1.0 - ab)

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {ep}/{epochs}")
        for x0, _ in pbar:
            x0 = x0.to(device)
            t = torch.randint(0, T, (x0.size(0),), device=device).long()
            noise = torch.randn_like(x0)
            xt = q_sample(x0, t, noise, sqrt_ab, sqrt_1mab)
            noise_pred = model(xt, t)
            loss = F.mse_loss(noise_pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.item()))

        # quick sample each epoch
        model.eval()
        x = torch.randn(16, 1, 28, 28, device=device)
        for i in reversed(range(T)):
            tt = torch.full((16,), i, device=device, dtype=torch.long)
            x = p_sample(model, x, tt, betas, alphas, ab, device)
        grid = make_grid((x + 1.0) / 2.0, nrow=4)
        save_image(grid, f"sample_epoch_{ep}.png")

    torch.save(model.state_dict(), ckpt)
    print(f"Saved: {ckpt}")


@torch.no_grad()
def sample(T=200, ckpt="diffusion_ddpm_mnist.pt", out="samples.png", n=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleDenoiser().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    betas = linear_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    ab = torch.cumprod(alphas, dim=0)

    x = torch.randn(n, 1, 28, 28, device=device)
    for i in tqdm(reversed(range(T)), total=T, desc="sampling"):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        x = p_sample(model, x, t, betas, alphas, ab, device)

    grid = make_grid((x + 1.0) / 2.0, nrow=int(math.sqrt(n)))
    save_image(grid, out)
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--ckpt", type=str, default="diffusion_ddpm_mnist.pt")
    args = ap.parse_args()

    if args.train:
        train(epochs=args.epochs, T=args.T, ckpt=args.ckpt)
    elif args.sample:
        sample(T=args.T, ckpt=args.ckpt)
    else:
        print("Use --train or --sample")


if __name__ == "__main__":
    main()

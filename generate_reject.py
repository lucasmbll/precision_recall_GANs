#!/usr/bin/env python3
# generate.py — DRS cohérent pour f-GAN (g = Forward-KL)
import os, json, argparse
import torch
from torchvision.utils import save_image
from model import Generator, Discriminator

def strip_module_keys(state_dict):
    return { (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items() }

@torch.no_grad()
def estimate_tau(G, D, device, n_warmup=20000, batch=2048, q=0.95, clamp_low=-10.0, clamp_high=10.0, zdim=100):
    """Estime tau = quantile q des log-ratios sur un gros lot de fakes."""
    logs = []
    remaining = n_warmup
    while remaining > 0:
        b = min(batch, remaining)
        z = torch.randn(b, zdim, device=device)
        x = G(z)                                        # [B, 784] dans [0,1]
        T = D(x).view(-1)                               # logits non bornés
        log_r = (T - 1.0).clamp(clamp_low, clamp_high)  # stabilité numérique
        logs.append(log_r)
        remaining -= b
    logs = torch.cat(logs, dim=0)
    return logs.quantile(q).item()

@torch.no_grad()
def save_batch_images(x_flat, out_dir, start_idx, to_rgb=False):
    """Sauve une batch d'images MNIST à partir d'un tenseur [B,784] en PNGs."""
    x = x_flat.view(-1, 1, 28, 28)
    if to_rgb:
        x = x.repeat(1, 3, 1, 1)
    n = x.size(0)
    for i in range(n):
        idx = start_idx + i
        save_image(x[i], os.path.join(out_dir, f"{idx:05d}.png"), value_range=(0, 1))

def main():
    ap = argparse.ArgumentParser(description="Generate images with Discriminative Rejection Sampling (DRS) for f-GAN (g=Forward-KL).")
    ap.add_argument("--out", type=str, default="samples", help="Output directory for accepted images")
    ap.add_argument("--checkpoints", type=str, default="checkpoints", help="Folder with G.pth / D.pth")
    ap.add_argument("--n", type=int, default=10000, help="Number of accepted images to save")
    ap.add_argument("--batch_size", type=int, default=2048, help="Batch size for proposal samples")
    ap.add_argument("--beta", type=float, default=8.0, help="Logistic sharpness β in p_acc = sigmoid(β(log r - τ))")
    ap.add_argument("--q", type=float, default=0.95, help="Quantile q to define τ on warm-up")
    ap.add_argument("--warmup", type=int, default=20000, help="Warm-up #fake samples to estimate τ")
    ap.add_argument("--clamp_low", type=float, default=-10.0, help="Clamp lower bound for log r")
    ap.add_argument("--clamp_high", type=float, default=10.0, help="Clamp upper bound for log r")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    ap.add_argument("--rgb", action="store_true", help="Save 3-channel PNGs instead of 1-channel")
    ap.add_argument("--zdim", type=int, default=100, help="Latent dimension for G")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # Load models
    g_path = os.path.join(args.checkpoints, "G.pth")
    d_path = os.path.join(args.checkpoints, "D.pth")
    if not os.path.exists(g_path) or not os.path.exists(d_path):
        raise FileNotFoundError(f"Missing checkpoints in {args.checkpoints} (need G.pth and D.pth)")

    mnist_dim = 784
    G = Generator(g_output_dim=mnist_dim).to(device)
    D = Discriminator(d_input_dim=mnist_dim).to(device)

    G.load_state_dict(strip_module_keys(torch.load(g_path, map_location=device)))
    D.load_state_dict(strip_module_keys(torch.load(d_path, map_location=device)))
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)
    G.eval(); D.eval()

    # Prepare output
    os.makedirs(args.out, exist_ok=True)

    # Warm-up to estimate tau
    print(f"Estimating τ with warm-up={args.warmup}, q={args.q}…")
    tau = estimate_tau(G, D, device, n_warmup=args.warmup, batch=args.batch_size,
                       q=args.q, clamp_low=args.clamp_low, clamp_high=args.clamp_high, zdim=args.zdim)
    print(f"Estimated τ (quantile {args.q:.2f}) = {tau:.4f}")

    # Generate with DRS
    print("Start Generating with DRS...")
    saved = 0
    total = 0
    log_every = 10

    with torch.no_grad():
        while saved < args.n:
            z = torch.randn(args.batch_size, args.zdim, device=device)
            x_fake = G(z) 
            #x_fake = (x_fake + 1) * 0.5 uncomment if tanh is used
            T = D(x_fake).view(-1)                      # logits
            log_r = (T - 1.0).clamp(args.clamp_low, args.clamp_high)
            p_acc = torch.sigmoid(args.beta * (log_r - tau))  # logistic DRS
            u = torch.rand_like(p_acc)
            mask = (u < p_acc)
            accepted = x_fake[mask]
            batch_saved = accepted.size(0)

            if batch_saved > 0:
                # clip to not exceed args.n
                to_save = min(batch_saved, args.n - saved)
                save_batch_images(accepted[:to_save], args.out, saved, to_rgb=args.rgb)
                saved += to_save

            total += x_fake.size(0)
            if (total // args.batch_size) % log_every == 0:
                acc_rate = 100.0 * (saved / total) if total > 0 else 0.0
                print(f"Images sauvegardées: {saved} / {args.n} | Taux d'acceptation (cumul): {acc_rate:.2f}%")

    final_acc = 100.0 * saved / max(1, total)
    print(f"Terminé. {saved} images sauvegardées. Taux d'acceptation final: {final_acc:.2f}%")

    # Sauver un petit méta log
    meta = {
        "n": args.n, "batch_size": args.batch_size, "beta": args.beta, "q": args.q,
        "warmup": args.warmup, "tau": tau, "clamp": [args.clamp_low, args.clamp_high],
        "seed": args.seed, "acceptance_rate": final_acc, "out": args.out
    }
    with open(os.path.join(args.out, "drs_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Méta sauvegardée dans {os.path.join(args.out, 'drs_meta.json')}")
    
if __name__ == "__main__":
    main()
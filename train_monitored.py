import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from model import Generator, Discriminator
from utils import D_train, G_train, save_models, D_train_soft_labels, D_train_soft_labels_noise_inputs
from evaluate import inception_features_from_dir
from pytorch_fid.fid_score import calculate_frechet_distance
from knn_precision_recall_pytorch import knn_precision_recall_features

def generate_and_save_samples(G, device, num_samples=10000, save_dir='temp_samples'):
    """Generate samples and save them to a directory for evaluation."""
    os.makedirs(save_dir, exist_ok=True)
    G.eval()
    n_samples = 0
    with torch.no_grad():
        while n_samples < num_samples:
            z = torch.randn(min(512, num_samples - n_samples), 100).to(device)
            x = G(z)
            x = x.reshape(-1, 28, 28)
            for k in range(x.shape[0]):
                if n_samples < num_samples:
                    torchvision.utils.save_image(x[k:k+1], os.path.join(save_dir, f'{n_samples}.png'))
                    n_samples += 1
    G.train()

def evaluate_gan(G, device, real_dir='real_samples', k=3):
    """Evaluate FID, Precision, and Recall."""
    # Generate samples
    temp_dir = 'temp_eval_samples'
    generate_and_save_samples(G, device, num_samples=10000, save_dir=temp_dir)
    
    # Extract features
    F_real = inception_features_from_dir(real_dir, batch_size=50, device=device,
                                         dims=2048, max_samples=10000)
    F_fake = inception_features_from_dir(temp_dir, batch_size=50, device=device,
                                         dims=2048, max_samples=10000)
    
    # Compute FID
    mu_real = np.mean(F_real, axis=0)
    sigma_real = np.cov(F_real, rowvar=False)
    mu_fake = np.mean(F_fake, axis=0)
    sigma_fake = np.cov(F_fake, rowvar=False)
    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    
    # Compute Precision/Recall
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
    state = knn_precision_recall_features(
        ref_features=F_real,
        eval_features=F_fake,
        nhood_sizes=[k],
        row_batch_size=10000,
        col_batch_size=50000,
        devices=devices
    )
    
    # Clean up temp samples
    import shutil
    shutil.rmtree(temp_dir)
    
    return fid_value, state['precision'][0], state['recall'][0]

def save_fixed_samples(G, fixed_z, epoch, save_dir='progress'):
    """Generate and save a 3x3 grid of samples using fixed latent vectors."""
    os.makedirs(save_dir, exist_ok=True)
    G.eval()
    with torch.no_grad():
        samples = G(fixed_z).reshape(-1, 1, 28, 28)
        # Normalize from [-1, 1] to [0, 1] for visualization
        samples = (samples + 1) / 2
        
        # Create 3x3 grid
        fig, axes = plt.subplots(3, 3, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i].cpu().squeeze(), cmap='gray')
            ax.axis('off')
        plt.suptitle(f'Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'samples_epoch_{epoch:03d}.png'), dpi=100)
        plt.close()
    G.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN on MNIST with periodic evaluation.')
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--lr_D", type=float, default=0.0001, help="Learning rate for discriminator.")
    parser.add_argument("--lr_G", type=float, default=0.0004, help="Learning rate for generator.")
    parser.add_argument("--batch_size", type=int, default=128, help="Size of mini-batches for SGD.")
    parser.add_argument("--eval_every", type=int, default=10, help="Evaluate every N epochs.")
    parser.add_argument("--gpus", type=int, default=-1, help="Number of GPUs to use (-1 for all available).")
    args = parser.parse_args()

    to_download = False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA")
        if args.gpus == -1:
            args.gpus = torch.cuda.device_count()
            print(f"Using {args.gpus} GPUs.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")

    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('progress', exist_ok=True)
    os.makedirs('real_samples', exist_ok=True)
    
    data_path = os.getenv('DATA')
    if data_path is None:
        data_path = "data"
        to_download = True

    # Data Pipeline
    print('Dataset loading...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_dataset = datasets.MNIST(root=data_path, train=True, transform=transform, download=to_download)
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    print('Dataset loaded.')
    
    # Save real samples for evaluation (do this once)
    if len(os.listdir('real_samples')) == 0:
        print("Saving real samples for evaluation...")
        for i, (img, _) in enumerate(train_dataset):
            if i >= 10000:
                break
            torchvision.utils.save_image(img, f'real_samples/{i}.png')
        print("Real samples saved.")

    # Model setup
    print('Model loading...')
    mnist_dim = 784
    G = Generator(g_output_dim=mnist_dim).to(device)
    D = Discriminator(mnist_dim).to(device)

    if args.gpus > 1:
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)
    print('Model loaded.')

    # Fixed latent vectors for visualization (9 samples for 3x3 grid)
    fixed_z = torch.randn(9, 100).to(device)

    # Loss and optimizers
    criterion = nn.BCELoss()
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr_G, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr_D, betas=(0.5, 0.999))

    # Metrics logging
    metrics_log = []

    print('Start training:')
    n_epoch = args.epochs
    for epoch in range(1, n_epoch + 1):
        d_sum, g_sum, n_batches = 0.0, 0.0, 0

        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim).to(device)
            noise_std = max(0.0, 0.1 * (1 - epoch / n_epoch))
            d_loss = D_train_soft_labels_noise_inputs(x, G, D, D_optimizer, criterion, device, noise_std=noise_std)
            g_loss = G_train(x, G, D, G_optimizer, criterion, device)

            d_sum += float(d_loss)
            g_sum += float(g_loss)
            n_batches += 1

        d_avg = d_sum / max(1, n_batches)
        g_avg = g_sum / max(1, n_batches)
        
        # Save fixed samples every epoch (lightweight)
        save_fixed_samples(G, fixed_z, epoch)
        
        # Evaluate and log
        if epoch % args.eval_every == 0:
            print(f"[Epoch {epoch:03d}] D_loss: {d_avg:.4f} | G_loss: {g_avg:.4f}")
            print(f"Evaluating at epoch {epoch}...")
            fid, precision, recall = evaluate_gan(G, device)
            print(f"FID: {fid:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
            
            metrics_log.append({
                'epoch': epoch,
                'fid': float(fid),  # Convert to native Python float
                'precision': float(precision),  # Convert to native Python float
                'recall': float(recall),  # Convert to native Python float
                'd_loss': d_avg,
                'g_loss': g_avg
            })
            
            save_models(G, D, 'checkpoints')
        else:
            print(f"[Epoch {epoch:03d}] D_loss: {d_avg:.4f} | G_loss: {g_avg:.4f}")

    # Save metrics to file
    import json
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics_log, f, indent=2)

    print('Training done.')
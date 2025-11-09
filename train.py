# train.py
import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from model import Generator, Discriminator
# Importer les NOUVELLES fonctions de l'algorithme 2023
from utils import D_train_auxiliary, G_train_primal, save_models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train f-GAN (2023 algorithm) on MNIST.')
    parser.add_argument("--lambda_", type=float, default=1.0, help="PR-divergence lambda (>1 precision, <1 recall).")
    parser.add_argument("--epochs", type=int, default=250, help="Number of epochs for training.")
    parser.add_argument("--lr_D", type=float, default=0.0003, help="Learning rate.")
    parser.add_argument("--lr_G", type=float, default=0.0005, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=128, help="Size of mini-batches for SGD.")
    parser.add_argument("--r1_gamma", type=float, default=5.0, help="R1 regularization gamma.")
    parser.add_argument("--noise_std", type=float, default=0.05, help="Standard deviation of instance noise.")
    parser.add_argument("--use_pearson", default=True, help="Use Pearson f-divergence instead of KL.")
    parser.add_argument("--gpus", type=int, default=-1, help="Number of GPUs to use (-1 for all available).")
    args = parser.parse_args()

    to_download=False
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
        
    os.makedirs('checkpoints', exist_ok=True)
    data_path = os.getenv('DATA')
    if data_path is None:
        data_path = "data"
        to_download = True

    # Data Pipeline
    print('Dataset loading...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(0.5, 0.5)  # uncomment if using tanh in G
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

    # Model setup
    print('Model loading...')
    mnist_dim = 784
    G = Generator(g_output_dim=mnist_dim).to(device)
    D = Discriminator(mnist_dim).to(device)

    if args.gpus > 1:
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)
    print('Model loaded.')



    # Loss and optimizers
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr_G, betas=(0.00, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr_D, betas=(0.00, 0.999))

    print('Start training:')
    n_epoch = args.epochs
    for epoch in range(1, n_epoch + 1):
        d_sum, g_sum, n_batches = 0.0, 0.0, 0

        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim).to(device)
            
            d_loss = D_train_auxiliary(
                x, G, D, D_optimizer, device,
                r1_gamma=args.r1_gamma,
                noise_std=args.noise_std,
                use_pearson=args.use_pearson
            )
            g_loss = G_train_primal(
                G, D, G_optimizer, device, x.shape[0],
                noise_std=args.noise_std,
                lambda_=args.lambda_,
                use_pearson=args.use_pearson
            )

            d_sum += float(d_loss)
            g_sum += float(g_loss)
            n_batches += 1

        d_avg = d_sum / max(1, n_batches)
        g_avg = g_sum / max(1, n_batches)
        print(f"[Epoch {epoch:03d}] D_loss (aux g): {d_avg:.4f} | G_loss (PR λ={args.lambda_}): {g_avg:.4f}")

        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')
            print(f"Models saved at epoch {epoch}.")

    print('Training done.')
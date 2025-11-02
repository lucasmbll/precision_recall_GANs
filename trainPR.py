import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from model import Generator, Discriminator, DiscriminatorPR
from utils import D_train, G_train, save_models, D_train_soft_labels, D_train_soft_labels_noise_inputs, D_train_PR, G_train_PR  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN on MNIST.')
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--lr_D", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--lr_G", type=float, default=0.0004, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=128, help="Size of mini-batches for SGD.")
    parser.add_argument("--gpus", type=int, default=-1, help="Number of GPUs to use (-1 for all available).")
    parser.add_argument("--pr_lambda", type=float, default=None,
                        help="Enable PR training with this λ (e.g., 0.2 for recall, 2.0 for precision).")
    args = parser.parse_args()

    to_download=False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
        print(f"Using device: CUDA")
        # Use all available GPUs if args.gpus is -1
        if args.gpus == -1:
            args.gpus = torch.cuda.device_count()
            print(f"Using {args.gpus} GPUs.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_type = "mps"
        print(f"Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        print(f"Using device: CPU")
        

    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
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
    test_dataset = datasets.MNIST(root=data_path, train=False, transform=transform, download=to_download)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,  # Use multiple workers for data loading
        pin_memory=True  # Faster data transfer to GPU
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print('Dataset loaded.')

    # Model setup
    print('Model loading...')
    mnist_dim = 784
    G = Generator(g_output_dim=mnist_dim).to(device)
    # Choose discriminator based on mode
    if args.pr_lambda is None:
        D = Discriminator(mnist_dim).to(device)     # BCE mode (sigmoid)
    else:
        D = DiscriminatorPR(mnist_dim).to(device)   # PR mode (raw score)

    # Wrap models in DataParallel if multiple GPUs are available
    if args.gpus > 1:
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)
    print('Model loaded.')

    # Loss and optimizers
    criterion = nn.BCELoss()
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr_G, betas=(0.5, 0.999))  # Changed from (0.0, 0.9)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr_D, betas=(0.5, 0.999))  # Changed from (0.0, 0.9)

    print('Start training:')
    n_epoch = args.epochs
    for epoch in range(1, n_epoch + 1):
        d_sum, g_sum, n_batches = 0.0, 0.0, 0

        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim).to(device)

            if args.pr_lambda is None:
                # ------- Original BCE path (unchanged) -------
                # d_loss = D_train(x, G, D, D_optimizer, criterion, device)
                noise_std = max(0.0, 0.1 * (1 - epoch / n_epoch))
                d_loss = D_train_soft_labels_noise_inputs(x, G, D, D_optimizer, criterion, device, noise_std=noise_std)
                g_loss = G_train(x, G, D, G_optimizer, criterion, device)
            else:
                # ------- PR path (paper’s approach) -------
                d_loss = D_train_PR(x, G, D, D_optimizer)                 # T: dual χ²
                g_loss = G_train_PR(x, G, D, G_optimizer, args.pr_lambda) # G: primal f_λ

            d_sum += float(d_loss); g_sum += float(g_loss); n_batches += 1

        print(f"[Epoch {epoch:03d}] D_loss: {d_sum/max(1,n_batches):.4f} | G_loss: {g_sum/max(1,n_batches):.4f}")

        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')

    print('Training done.')

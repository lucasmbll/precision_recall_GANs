import argparse
import torch
from pytorch_fid.fid_score import calculate_frechet_distance
from knn_precision_recall_pytorch import knn_precision_recall_features
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from pytorch_fid.inception import InceptionV3
import numpy as np

class _FlatImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.paths = []
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        for fn in os.listdir(root):
            if os.path.splitext(fn)[1].lower() in exts:
                self.paths.append(os.path.join(root, fn))
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img) if self.transform else img

def _inception_model(dims=2048, device='cpu'):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], resize_input=True, normalize_input=False).to(device)
    model.eval()
    return model


@torch.no_grad()
def inception_features_from_dir(img_dir, batch_size=50, device='cpu', dims=2048, num_workers=4, max_samples=None):
    tfm = transforms.Compose([
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Add normalization
    ])
    ds = _FlatImageFolder(img_dir, transform=tfm)
    if max_samples is not None and max_samples < len(ds):
        idx = np.random.choice(len(ds), size=max_samples, replace=False)
        ds = torch.utils.data.Subset(ds, idx)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    model = _inception_model(dims, device)
    feats = []
    for x in dl:
        x = x.to(device, non_blocking=True)
        f = model(x)[0].view(x.size(0), -1)  # pool3 -> (B, 2048)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0).numpy().astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID between real and fake samples using pytorch-fid.")
    parser.add_argument("--k", type=int, default=3) # k = 3 is a robust choice that avoids saturating the values (ref paper)
    parser.add_argument("--row_bs", type=int, default=10000)
    parser.add_argument("--col_bs", type=int, default=50000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Extracting Inception features...")
    F_real = inception_features_from_dir('real_samples', batch_size=50, device=device,
                                         dims=2048, max_samples=10000)
    F_fake = inception_features_from_dir('samples', batch_size=50, device=device,
                                         dims=2048, max_samples=10000)

    print("Calculating FID...")
    mu_real = np.mean(F_real, axis=0)
    sigma_real = np.cov(F_real, rowvar=False)
    mu_fake = np.mean(F_fake, axis=0)
    sigma_fake = np.cov(F_fake, rowvar=False)
    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    print(f"FID Score: {fid_value:.4f}")
    
    # Auto: all GPUs if available
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]

    state = knn_precision_recall_features(
        ref_features=F_real,
        eval_features=F_fake,
        nhood_sizes=[args.k],
        row_batch_size=args.row_bs,
        col_batch_size=args.col_bs,
        devices=devices
    )
    k = int(state["k"][0])
    print(f"Precision with k = {k}: {state['precision'][0]:.4f}")
    print(f"Recall with k = {k}   : {state['recall'][0]:.4f}")
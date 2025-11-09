import argparse
import torch
from time import time
from pytorch_fid.fid_score import calculate_frechet_distance
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from pytorch_fid.inception import InceptionV3
import numpy as np
from knn_precision_recall_pytorch import DistanceBlock

# ------------------------------------------------------------------------
class ManifoldEstimator:
    """
    Estimates the manifold radii (k-NN squared distances) for a reference feature set
    using row/col batching and a DistanceBlock for multi-GPU speed.
    """
    def __init__(self,
                 distance_block: DistanceBlock,
                 features: np.ndarray,
                 row_batch_size: int = 25000,
                 col_batch_size: int = 50000,
                 nhood_sizes = [3],
                 clamp_to_percentile: float = None,
                 eps: float = 1e-5):
        """
        features: np.float32 array of shape (N, D)
        nhood_sizes: list of k values (e.g., [3]) for k-NN radii
        """
        assert isinstance(features, np.ndarray)
        assert features.ndim == 2
        features = features.astype(np.float32, copy=False)

        num_images = features.shape[0]
        self.nhood_sizes = list(nhood_sizes)
        self.num_nhoods = len(self.nhood_sizes)
        self.eps = eps
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._ref_features = features
        self._distance_block = distance_block

        # D: (N, len(nhood_sizes)) holds squared k-NN distances per sample
        self.D = np.zeros((num_images, self.num_nhoods), dtype=np.float32)

        # Work buffer for one row-batch against all columns
        distance_batch = np.zeros((row_batch_size, num_images), dtype=np.float32)
        # Sequence indices needed by np.partition (up to max(k)+1 to include self-dist=0)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        # Compute chunked distances and k-NN radii
        for begin1 in range(0, num_images, row_batch_size):
            end1 = min(begin1 + row_batch_size, num_images)
            row_batch = features[begin1:end1]  # (B, D)

            # Fill whole row chunk by scanning columns in blocks
            for begin2 in range(0, num_images, col_batch_size):
                end2 = min(begin2 + col_batch_size, num_images)
                col_batch = features[begin2:end2]  # (C, D)
                distance_batch[0:end1-begin1, begin2:end2] = \
                    self._distance_block.pairwise_distances(row_batch, col_batch)

            # For each row, take k-th smallest (squared) distances ignoring self via +1 indexing
            # Matching the TF behavior: partition at indices seq, then pick entries at k
            # (k=0 -> self, k=1 -> nearest neighbor, etc.)
            part = np.partition(distance_batch[0:end1-begin1, :], seq, axis=1)
            self.D[begin1:end1, :] = part[:, self.nhood_sizes]

        # Optional pruning: set too-large radii to zero (as in the TF code)
        if clamp_to_percentile is not None:
            max_distances = np.percentile(self.D, clamp_to_percentile, axis=0).astype(np.float32)
            mask = self.D > max_distances[None, :]
            self.D[mask] = 0.0

    def evaluate(self, eval_features: np.ndarray,
                 return_realism: bool = False,
                 return_neighbors: bool = False):
        """
        Check if eval features fall inside *any* ball of the manifold (per k in nhood_sizes).
        Returns a (N_eval, len(nhood_sizes)) int array of 0/1.
        Optionally returns realism score and nearest neighbor indices (like TF code).
        """
        assert isinstance(eval_features, np.ndarray)
        eval_features = eval_features.astype(np.float32, copy=False)

        num_eval_images = eval_features.shape[0]
        num_ref_images  = self.D.shape[0]
        B = self.row_batch_size

        distance_batch = np.zeros((B, num_ref_images), dtype=np.float32)
        batch_predictions = np.zeros((num_eval_images, self.num_nhoods), dtype=np.int32)
        max_realism_score = np.zeros((num_eval_images,), dtype=np.float32)
        nearest_indices   = np.zeros((num_eval_images,), dtype=np.int32)

        for begin1 in range(0, num_eval_images, B):
            end1 = min(begin1 + B, num_eval_images)
            feature_batch = eval_features[begin1:end1]  # (b, D)

            # distances from this feature batch to all reference samples (chunked by columns)
            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = self._ref_features[begin2:end2]
                distance_batch[0:end1-begin1, begin2:end2] = \
                    self._distance_block.pairwise_distances(feature_batch, ref_batch)

            # Membership test: for each eval sample and each k, does it fit inside ANY ref ball?
            # Compare (b, Nref) against (Nref, num_nhoods) radii via broadcasting.
            # We want: exists j s.t. dist[i, j] <= D[j, k]
            # shape align: (b, Nref, 1) <= (Nref, num_nhoods) -> (b, Nref, num_nhoods)
            samples_in_manifold = distance_batch[0:end1-begin1, :, None] <= self.D[None, :, :]
            # any over ref axis -> (b, num_nhoods)
            batch_predictions[begin1:end1, :] = np.any(samples_in_manifold, axis=1).astype(np.int32)

            # Optional realism & nearest neighbor (mimic TF)
            # realism ~ max over j of D[j, 0] / (dist[i, j] + eps)
            # nearest index ~ argmin dist[i, j]
            d_sub = distance_batch[0:end1-begin1, :]  # (b, Nref)
            denom = d_sub + self.eps
            max_realism_score[begin1:end1] = np.max(self.D[:, 0][None, :] / denom, axis=1)
            nearest_indices[begin1:end1]   = np.argmin(d_sub, axis=1)

        if return_realism and return_neighbors:
            return batch_predictions, max_realism_score, nearest_indices
        elif return_realism:
            return batch_predictions, max_realism_score
        elif return_neighbors:
            return batch_predictions, nearest_indices
        return batch_predictions

# ------------------------------------------------------------------------
def knn_precision_recall_features(
        ref_features: np.ndarray,
        eval_features: np.ndarray,
        nhood_sizes = [3],
        row_batch_size: int = 10000,
        col_batch_size: int = 50000,
        devices=None):
    """
    Computes Kynkäänniemi et al. (NeurIPS'19) k-NN Precision/Recall on features.
    - ref_features: (N_ref, D) real/reference features (float32)
    - eval_features: (N_eval, D) generated features (float32)
    - nhood_sizes: list of k values (default [3])
    - devices: e.g., ['cuda:0','cuda:1'] or None (auto)
    Returns: dict with 'precision' and 'recall' as np.float32 arrays of shape (len(nhood_sizes),)
    """
    ref_features  = ref_features.astype(np.float32, copy=False)
    eval_features = eval_features.astype(np.float32, copy=False)
    num_features = ref_features.shape[1]
    assert eval_features.shape[1] == num_features

    distance_block = DistanceBlock(num_features, devices=devices)
    ref_manifold  = ManifoldEstimator(distance_block, ref_features,
                                      row_batch_size=row_batch_size,
                                      col_batch_size=col_batch_size,
                                      nhood_sizes=nhood_sizes)
    eval_manifold = ManifoldEstimator(distance_block, eval_features,
                                      row_batch_size=row_batch_size,
                                      col_batch_size=col_batch_size,
                                      nhood_sizes=nhood_sizes)

    precision_mat = ref_manifold.evaluate(eval_features)  # (N_eval, len(k))
    recall_mat    = eval_manifold.evaluate(ref_features)  # (N_ref,  len(k))

    state = {
        'precision': precision_mat.mean(axis=0).astype(np.float32),
        'recall':    recall_mat.mean(axis=0).astype(np.float32),
        'k':         np.array(nhood_sizes, dtype=np.int32),
    }
    return state


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
    model = InceptionV3([block_idx], resize_input=False, normalize_input=True).to(device)
    model.eval()
    return model


@torch.no_grad()
def inception_features_from_dir(img_dir, batch_size=50, device='cpu', dims=2048, num_workers=4, max_samples=None):
    tfm = transforms.Compose([
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
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
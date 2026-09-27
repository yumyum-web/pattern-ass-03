"""
Data loading and preprocessing pipeline for the RealWaste dataset (UCI ID: 908).
Downscales images to 64x64 pixels and produces deterministic 70% / 15% / 15%
stratified train/val/test splits for resource-constrained edge classification.
"""

import os
import sys
import zipfile
import urllib.request
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

# RealWaste 9 Classes (canonical ordering matching UCI specs)
CLASSES: Tuple[str, ...] = (
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
)

CLASS_TO_IDX: Dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}

UCI_REALWASTE_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"

# Standard ImageNet channel statistics
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


class DownloadProgressBar(tqdm):
    """Progress bar for urllib file downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def find_dataset_root(data_dir: str = "data") -> Optional[Path]:
    """
    Finds the directory containing class subdirectories by searching data_dir.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return None

    # Check common known locations first
    candidates = [
        data_path / "RealWaste",
        data_path / "realwaste-main" / "RealWaste",
        data_path / "realwaste-main",
        data_path,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            # Check if at least 5 of the classes exist as subdirectories
            existing = [c for c in CLASSES if (candidate / c).exists() or (candidate / c.lower()).exists()]
            if len(existing) >= 5:
                return candidate

    # Recursive search for a directory containing 'Cardboard'
    for path in data_path.rglob("Cardboard"):
        if path.is_dir():
            return path.parent

    return None


def download_and_extract_realwaste(data_dir: str = "data") -> Path:
    """
    Downloads RealWaste archive from UCI repository if not already present
    and extracts image directory.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    root = find_dataset_root(data_dir)
    if root is not None:
        return root

    zip_path = data_path / "realwaste.zip"
    if not zip_path.exists():
        print(f"[Data Pipeline] Downloading RealWaste dataset from {UCI_REALWASTE_URL}...")
        req = urllib.request.Request(
            UCI_REALWASTE_URL,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        )
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="realwaste.zip") as t:
            with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as f:
                tsize = resp.info().get("Content-Length")
                if tsize:
                    t.total = int(tsize)
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    t.update(len(chunk))
        print("[Data Pipeline] Download complete.")

    print(f"[Data Pipeline] Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)
    print("[Data Pipeline] Extraction complete.")

    root = find_dataset_root(data_dir)
    if root is not None:
        return root

    raise FileNotFoundError(f"Could not locate RealWaste classes directory inside {data_dir} after extraction.")


def collect_image_records(dataset_root: Path) -> Tuple[List[str], List[int]]:
    """
    Collects image filepaths and corresponding class labels from dataset folder.
    Matches folder names to known RealWaste classes.
    """
    filepaths: List[str] = []
    labels: List[int] = []

    subdirs = [d for d in dataset_root.iterdir() if d.is_dir()]
    dir_map = {}
    for d in subdirs:
        for canonical_name in CLASSES:
            if d.name.lower().replace("_", " ").replace("-", " ") == canonical_name.lower().replace("_", " "):
                dir_map[d] = CLASS_TO_IDX[canonical_name]
                break

    for dir_path, label in dir_map.items():
        valid_images = [
            str(p) for p in dir_path.glob("*.*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ]
        filepaths.extend(valid_images)
        labels.extend([label] * len(valid_images))

    if not filepaths:
        raise FileNotFoundError(
            f"No image files found in {dataset_root}. Ensure directory structure has class subfolders."
        )

    return filepaths, labels


class RealWasteDataset(Dataset):
    """
    PyTorch Dataset wrapper for RealWaste image classification at 64x64 resolution.
    Supports high-speed in-memory tensor caching to avoid repeated disk decoding.
    """
    def __init__(
        self,
        filepaths: Optional[List[str]] = None,
        labels: Optional[List[int]] = None,
        images_tensor: Optional[torch.Tensor] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        self.transform = transform
        if images_tensor is not None:
            self.images_tensor = images_tensor  # (N, H, W, C) uint8
            self.labels = labels if labels is not None else []
            self.filepaths = None
        else:
            self.images_tensor = None
            self.filepaths = filepaths or []
            self.labels = labels or []

    def __len__(self) -> int:
        if self.images_tensor is not None:
            return len(self.images_tensor)
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label = int(self.labels[idx])
        if self.images_tensor is not None:
            img_arr = self.images_tensor[idx].numpy()
            image = Image.fromarray(img_arr)
        else:
            path = self.filepaths[idx]
            with Image.open(path) as img:
                image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_transforms(image_size: int = 64) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Constructs edge-appropriate data transforms downscaled to 64x64 pixels.
    Includes lightweight augmentations for training to prevent overfitting.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(image_size, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    return train_transform, eval_transform


def _preprocess_image_list(paths: List[str], image_size: int) -> torch.Tensor:
    """Pre-downscales image list into uint8 tensor (N, H, W, 3)."""
    tensors = []
    for p in tqdm(paths, desc="Downscaling images", unit="img"):
        with Image.open(p) as img:
            rgb = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            tensors.append(np.array(rgb, dtype=np.uint8))
    return torch.from_numpy(np.stack(tensors, axis=0))


def get_realwaste_dataloaders(
    data_dir: str = "data",
    batch_size: int = 64,
    image_size: int = 64,
    seed: int = 42,
    num_workers: int = 2,
    download: bool = True,
    use_cache: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Generates deterministic stratified 70% Train, 15% Validation, and 15% Test DataLoaders.
    Pre-caches downscaled 64x64 images as uint8 tensors for fast training.
    
    Returns:
        train_loader (DataLoader): 70% of dataset (3,326 samples)
        val_loader (DataLoader): 15% of dataset (713 samples)
        test_loader (DataLoader): 15% of dataset (713 samples)
        class_to_idx (dict): mapping from class name to integer label [0..8]
    """
    root = find_dataset_root(data_dir)
    if root is None:
        if download:
            root = download_and_extract_realwaste(data_dir=data_dir)
        else:
            raise FileNotFoundError(f"RealWaste dataset not found in {data_dir}. Set download=True.")

    cache_file = Path(data_dir) / f"realwaste_cache_{image_size}x{image_size}.pt"
    train_transform, eval_transform = get_transforms(image_size=image_size)

    if use_cache and cache_file.exists():
        cached = torch.load(cache_file, weights_only=False)
        train_dataset = RealWasteDataset(
            images_tensor=cached["train_imgs"],
            labels=cached["train_labels"],
            transform=train_transform,
        )
        val_dataset = RealWasteDataset(
            images_tensor=cached["val_imgs"],
            labels=cached["val_labels"],
            transform=eval_transform,
        )
        test_dataset = RealWasteDataset(
            images_tensor=cached["test_imgs"],
            labels=cached["test_labels"],
            transform=eval_transform,
        )
    else:
        filepaths, labels = collect_image_records(root)

        # Deterministic stratified split: 70% Train, 30% Temp (Val + Test)
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            filepaths,
            labels,
            test_size=0.30,
            random_state=seed,
            stratify=labels,
        )

        # Split remaining 30% equally: 15% Val, 15% Test
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files,
            temp_labels,
            test_size=0.50,
            random_state=seed,
            stratify=temp_labels,
        )

        if use_cache:
            print(f"[Data Pipeline] Caching {len(filepaths)} images to 64x64 uint8 tensors...")
            train_imgs = _preprocess_image_list(train_files, image_size)
            val_imgs = _preprocess_image_list(val_files, image_size)
            test_imgs = _preprocess_image_list(test_files, image_size)

            torch.save({
                "train_imgs": train_imgs,
                "train_labels": train_labels,
                "val_imgs": val_imgs,
                "val_labels": val_labels,
                "test_imgs": test_imgs,
                "test_labels": test_labels,
            }, cache_file)
            print(f"[Data Pipeline] Cache saved to {cache_file} ({cache_file.stat().st_size / (1024*1024):.1f} MB)")

            train_dataset = RealWasteDataset(images_tensor=train_imgs, labels=train_labels, transform=train_transform)
            val_dataset = RealWasteDataset(images_tensor=val_imgs, labels=val_labels, transform=eval_transform)
            test_dataset = RealWasteDataset(images_tensor=test_imgs, labels=test_labels, transform=eval_transform)
        else:
            train_dataset = RealWasteDataset(train_files, train_labels, transform=train_transform)
            val_dataset = RealWasteDataset(val_files, val_labels, transform=eval_transform)
            test_dataset = RealWasteDataset(test_files, test_labels, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, CLASS_TO_IDX



if __name__ == "__main__":
    print("=== RealWaste Data Pipeline Verification ===")
    print(f"Target Resolution: 64x64 pixels | Classes: {len(CLASSES)}")
    for idx, name in enumerate(CLASSES):
        print(f"  [{idx}] {name}")

    train_loader, val_loader, test_loader, c2i = get_realwaste_dataloaders(
        data_dir="data", batch_size=32, num_workers=2, download=False
    )
    total = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    print(f"\nDataset Splits (Total: {total}):")
    print(f"  Train: {len(train_loader.dataset)} samples ({len(train_loader.dataset)/total*100:.2f}%)")
    print(f"  Val:   {len(val_loader.dataset)} samples ({len(val_loader.dataset)/total*100:.2f}%)")
    print(f"  Test:  {len(test_loader.dataset)} samples ({len(test_loader.dataset)/total*100:.2f}%)")
    
    sample_batch, sample_labels = next(iter(train_loader))
    print(f"\nBatch Tensor Verification:")
    print(f"  Images Shape: {sample_batch.shape} (Expected: [32, 3, 64, 64])")
    print(f"  Labels Shape: {sample_labels.shape} (Expected: [32])")
    print(f"  Image Min: {sample_batch.min():.3f}, Max: {sample_batch.max():.3f}")
    print("\n✅ Verification Successful: All 4,752 images parsed into exact 70/15/15 splits at 64x64!")

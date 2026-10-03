#!/usr/bin/env python3
"""
Dataset download and preprocessing helper for paper reproduction.

Usage:
    python download_data.py --dataset <name> [--output-dir data/]

Supported datasets:
    - cifar10, cifar100: Auto-download via torchvision
    - mnist, fashion-mnist: Auto-download via torchvision
    - imdb, ag_news, sst2: Auto-download via HuggingFace datasets
    - coco: Instructions for manual download
    - imagenet: Instructions for manual download
"""

import argparse
import os
import sys


def download_torchvision_dataset(name, output_dir):
    """Download torchvision datasets."""
    try:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
    except ImportError:
        print("ERROR: torchvision not installed. Run: pip install torchvision")
        sys.exit(1)

    dataset_map = {
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
        "mnist": datasets.MNIST,
        "fashion-mnist": datasets.FashionMNIST,
    }

    if name not in dataset_map:
        print(f"Unknown torchvision dataset: {name}")
        return False

    print(f"Downloading {name} to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    ds = dataset_map[name](root=output_dir, train=True, download=True, transform=transform)
    ds_test = dataset_map[name](root=output_dir, train=False, download=True, transform=transform)

    print(f"Done! Train: {len(ds)}, Test: {len(ds_test)}")
    return True


def download_hf_dataset(name, output_dir):
    """Download HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets not installed. Run: pip install datasets")
        sys.exit(1)

    hf_map = {
        "imdb": "imdb",
        "ag_news": "ag_news",
        "sst2": "sst2",
        "squad": "squad",
        "mnli": "glue",
    }

    hf_name = hf_map.get(name)
    if not hf_name:
        print(f"Unknown HF dataset: {name}")
        return False

    print(f"Downloading {name} from HuggingFace...")
    os.makedirs(output_dir, exist_ok=True)

    if name == "mnli":
        ds = load_dataset(hf_name, "mnli")
    else:
        ds = load_dataset(hf_name)

    print(f"Done! Splits: {list(ds.keys())}")
    for split, data in ds.items():
        print(f"  {split}: {len(data)} examples")
    return True


def print_manual_instructions(name):
    """Print manual download instructions for large datasets."""
    instructions = {
        "imagenet": """
ImageNet (ILSVRC 2012) requires manual download due to access restrictions.

1. Register at https://image-net.org/
2. Download ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar
3. Extract to data/imagenet/train/ and data/imagenet/val/
4. Organize val/ by synset folders (use valprep.sh from the website)

Expected structure:
  data/imagenet/
    train/
      n01440764/
        n01440764_10026.JPEG
        ...
      n01443537/
        ...
    val/
      n01440764/
        ILSVRC2012_val_00000293.JPEG
        ...
""",
        "coco": """
COCO requires manual download.

1. Download from https://cocodataset.org/#download
2. For detection:
   - wget http://images.cocodataset.org/zips/train2017.zip
   - wget http://images.cocodataset.org/zips/val2017.zip
   - wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
3. Extract to data/coco/

Expected structure:
  data/coco/
    train2017/
    val2017/
    annotations/
""",
        "ade20k": """
ADE20K requires registration.

1. Register at http://sceneparsing.csail.mit.edu/
2. Download the dataset
3. Extract to data/ade20k/
""",
    }

    if name in instructions:
        print(instructions[name])
    else:
        print(f"No manual instructions available for: {name}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for paper reproduction")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    args = parser.parse_args()

    name = args.dataset.lower()
    output_dir = args.output_dir

    # Try torchvision first
    if download_torchvision_dataset(name, output_dir):
        return

    # Try HuggingFace
    if download_hf_dataset(name, output_dir):
        return

    # Manual instructions
    print_manual_instructions(name)


if __name__ == "__main__":
    main()

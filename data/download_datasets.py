"""
download_datasets.py
====================
CLI utility for acquiring standard civil infrastructure inspection datasets.

Datasets:
1. DeepCrack (~150 MB): Paired RGB crack images and binary segmentation masks.
2. SDNET2018 (~550 MB): Utah State Univ. concrete bridge decks, walls, pavement patches.
   Includes ~47,400 uncracked (U/) samples for Stage 2 negative-only training.
3. METU Concrete Cracks (~235 MB): 40,000 RGB 227x227 concrete images (20k crack, 20k no-crack).
4. CODEBRIM (~4.2 GB): Multi-target multi-label concrete bridge defect dataset.
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import shutil

DATASET_CATALOG = {
    "deepcrack": {
        "name": "DeepCrack",
        "description": "537 paired RGB crack images + ground-truth binary masks for Stage 2 pixel AUROC evaluation.",
        "url": "https://github.com/yhlscut/DeepCrack/archive/refs/heads/master.zip",
        "format": "zip",
        "auto_downloadable": True,
        "citation": "Liu et al., 'DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation', Neurocomputing 2019."
    },
    "sdnet2018": {
        "name": "SDNET2018",
        "description": "56,000+ concrete bridge deck, wall, and pavement patches for Stage 1 & Stage 2 negative-only training.",
        "url": "https://digitalcommons.usu.edu/context/all_datasets/article/1047/type/native/viewcontent",
        "format": "zip",
        "auto_downloadable": False,
        "landing_page": "https://digitalcommons.usu.edu/all_datasets/48/",
        "citation": "Dorafshan et al., 'SDNET2018: An annotated image dataset for training, validation, and benchmarking of artificial neural networks for concrete crack detection', Utah State University 2018."
    },
    "metu_cracks": {
        "name": "METU Concrete Crack Images",
        "description": "40,000 RGB 227x227 concrete surface images (20k negative healthy, 20k positive cracked).",
        "url": "https://data.mendeley.com/public-files/datasets/5y9wdsg2zt/files/8a70d8a5-b17a-496b-944d-5c8e43e2de3a/file_downloaded",
        "format": "zip",
        "auto_downloadable": False,
        "landing_page": "https://data.mendeley.com/datasets/5y9wdsg2zt/2",
        "citation": "Özgenel, Ç. F. (2019). Concrete Crack Images for Classification, Mendeley Data, V2."
    },
    "codebrim": {
        "name": "CODEBRIM",
        "description": "Multi-target multi-label civil defect dataset (cracks, spalling, efflorescence, exposed bars, corrosion stains).",
        "url": "https://zenodo.org/api/records/2620293/files-archive",
        "format": "zip",
        "auto_downloadable": False,
        "landing_page": "https://zenodo.org/records/2620293",
        "citation": "Mundt et al., 'Meta-learning for multi-task inspection of civil structures', ICIP 2019."
    }
}


def download_file(url: str, dest_path: str):
    """Downloads a file with dynamic percentage progress display."""
    print(f"[*] Connecting to: {url}")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            mb_downloaded = (count * block_size) / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r    Downloading: [{percent:3d}%] {mb_downloaded:.1f}/{mb_total:.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    print("\n[+] Download complete.")


def handle_dataset(key: str, out_dir: str):
    info = DATASET_CATALOG.get(key)
    if not info:
        print(f"[!] Unknown dataset key: {key}")
        return

    print("=" * 70)
    print(f"Dataset: {info['name']}")
    print(f"Role:    {info['description']}")
    print(f"Source:  {info['citation']}")
    print("=" * 70)

    target_dir = os.path.join(out_dir, key)
    os.makedirs(target_dir, exist_ok=True)

    if not info["auto_downloadable"]:
        print(f"[i] Dataset requires direct portal access:")
        print(f"    Portal URL: {info['landing_page']}")
        print(f"    Direct URL: {info['url']}")
        print(f"\nManual Installation Steps:")
        print(f"    1. Download the archive from the portal above.")
        print(f"    2. Extract files into: {os.path.abspath(target_dir)}")
        print(f"    3. Pass --data-dir {os.path.abspath(target_dir)} to training scripts.")
        return

    archive_path = os.path.join(target_dir, f"{key}.zip")
    try:
        download_file(info["url"], archive_path)
        print(f"[*] Extracting {archive_path}...")
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print(f"[+] Successfully extracted to: {target_dir}")
        os.remove(archive_path)
    except Exception as e:
        print(f"[!] Auto-download failed: {e}")
        print(f"    Please manually download from {info['url']} into {target_dir}")


def main():
    parser = argparse.ArgumentParser(description="Civil Inspection Dataset Downloader")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["deepcrack", "sdnet2018", "metu_cracks", "codebrim", "all"],
        help="Dataset identifier to acquire.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/raw",
        help="Destination directory for dataset archives and extractions.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.dataset == "all":
        for k in DATASET_CATALOG:
            handle_dataset(k, args.out_dir)
    else:
        handle_dataset(args.dataset, args.out_dir)


if __name__ == "__main__":
    main()

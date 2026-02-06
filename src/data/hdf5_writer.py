#(Build HDF5 from raw images & splits)
import h5py
import cv2
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
SPLITS_DIR = Path("data/splits")
HDF5_PATH = Path("data/processed/dataset.h5")

def build_hdf5():
    image_paths, labels = [], []
    for split_file in ["training.txt", "validation.txt", "test.txt"]:
        with open(SPLITS_DIR / split_file) as f:
            for line in f:
                parts = line.strip().split()
                path = RAW_DIR / parts[0]
                label = list(map(float, parts[1:]))
                image_paths.append(path)
                labels.append(label)
    
    num_samples = len(image_paths)
    h, w = 480, 640

    with h5py.File(HDF5_PATH, "w") as f:
        f.create_dataset("images", (num_samples, h, w, 3), dtype="uint8")
        f.create_dataset("labels", (num_samples, len(labels[0])), dtype="float32")
        for i, path in enumerate(image_paths):
            img = cv2.imread(str(path))
            img = cv2.resize(img, (w, h))
            f["images"][i] = img
            f["labels"][i] = labels[i]

    print(f"HDF5 dataset created at {HDF5_PATH}")

if __name__ == "__main__":
    build_hdf5()

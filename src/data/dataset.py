#(Lazy-loading HDF5 dataset)
import h5py
import torch
from torch.utils.data import Dataset
from .transforms import RandomBrightness, ToTensor

class IndustrialImageDataset(Dataset):
    def __init__(self, hdf5_path, transform=None):
        self.hdf5_path = hdf5_path
        self.transform = transform
        with h5py.File(self.hdf5_path, 'r') as f:
            self.length = len(f['images'])
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as f:
            image = f['images'][idx]
            label = f['labels'][idx]
        
        if self.transform:
            for t in self.transform:
                image = t(image)
        
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

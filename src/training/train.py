import yaml
import torch
from torch.utils.data import DataLoader
from src.data.dataset import IndustrialImageDataset
from src.data.transforms import RandomBrightness, ToTensor
from src.models.model import SimplePoseModel
from src.training.trainer import train_one_epoch

# Load config
with open("configs/train.yaml") as f:
    cfg = yaml.safe_load(f)

# Dataset
transforms = [RandomBrightness(), ToTensor()]
dataset = IndustrialImageDataset(cfg["dataset"]["hdf5_path"], transform=transforms)
loader = DataLoader(dataset, batch_size=cfg["dataset"]["batch_size"], num_workers=cfg["dataset"]["num_workers"])

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimplePoseModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
loss_fn = torch.nn.MSELoss()

# Training loop
for epoch in range(cfg["training"]["epochs"]):
    loss = train_one_epoch(model, loader, optimizer, loss_fn, device=device)
    print(f"Epoch {epoch+1}/{cfg['training']['epochs']}, Loss: {loss:.4f}")

# Save model
torch.save(model.state_dict(), cfg["training"]["save_path"] + "/model.pth")


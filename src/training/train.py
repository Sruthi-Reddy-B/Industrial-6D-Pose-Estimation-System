'''import yaml
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

'''
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from src.data.dataset import IndustrialImageDataset
from src.data.transforms import RandomBrightness, ToTensor
from src.models.model import SimplePoseModel
from src.training.trainer import train_one_epoch, validate

def main():
    # Load config
    with open("configs/train.yaml") as f:
        cfg = yaml.safe_load(f)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset with on-the-fly augmentation
    transforms = [RandomBrightness(), ToTensor()]
    dataset = IndustrialImageDataset(
        cfg["dataset"]["hdf5_path"], 
        transform=transforms
    )
    
    # Create dataloader
    # Note: num_workers=0 for simplicity. In production, use 4-8 workers
    # for faster data loading with multiprocessing
    loader = DataLoader(
        dataset, 
        batch_size=cfg["dataset"]["batch_size"], 
        num_workers=0,  # Single process for debugging/simple setups
        shuffle=True
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Initialize model
    model = SimplePoseModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    loss_fn = torch.nn.MSELoss()
    
    # Create save directory
    save_path = Path(cfg["training"]["save_path"])
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(cfg["training"]["epochs"]):
        train_loss = train_one_epoch(model, loader, optimizer, loss_fn, device=device)
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']}, Loss: {train_loss:.4f}")
    
    # Save model
    model_path = save_path / "model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Save config for reproducibility
    config_path = save_path / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f)
    print(f"✓ Config saved to {config_path}")

if __name__ == "__main__":
    main()

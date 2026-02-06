import torch
from torch.utils.data import DataLoader
from datasets.lazy_dataset import LazyImageDataset
from augmentations.transforms import train_transforms
from models.detector import SimpleDetector

# Configs (example)
image_dir = 'data/images'
split_file = 'data/splits/train.txt'
batch_size = 16
num_epochs = 5

# Dataset and DataLoader
dataset = LazyImageDataset(image_dir, split_file, transform=train_transforms)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = SimpleDetector()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()  # placeholder

# Training loop
for epoch in range(num_epochs):
    for imgs in dataloader:
        outputs = model(imgs.float())
        loss = criterion(outputs, outputs)  # dummy loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

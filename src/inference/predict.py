import torch
from src.models.model import SimplePoseModel

def predict(model_path, image_tensor, device='cpu'):
    model = SimplePoseModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        pred = model(image_tensor.unsqueeze(0).to(device))
    return pred.squeeze(0)


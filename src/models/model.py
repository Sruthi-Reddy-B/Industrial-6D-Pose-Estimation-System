import torch
import torch.nn as nn

class SimplePoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(480*640*3, 256),
            nn.ReLU(),
            nn.Linear(256, 6)  # 6D pose output
        )
    
    def forward(self, x):
        return self.net(x)


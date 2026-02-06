'''import torch
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
'''
import torch
import torch.nn as nn

class SimplePoseModel(nn.Module):
    """
    Simple CNN baseline for 6D pose estimation.
    
    Architecture: 3 conv blocks + global pooling + FC head
    This is a minimal but reasonable approach for computer vision.
    For production, use pretrained ResNet/EfficientNet backbones.
    """
    
    def __init__(self):
        super().__init__()
        
        # Feature extraction: 3 conv blocks with downsampling
        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 480x640 -> 240x320
            
            # Block 2: 32 -> 64 channels  
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 240x320 -> 120x160
            
            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 120x160 -> 60x80
        )
        
        # Global pooling to handle variable spatial sizes
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Regression head for 6D pose
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # 6D output: [x, y, z, roll, pitch, yaw]
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.head(x)
        return x

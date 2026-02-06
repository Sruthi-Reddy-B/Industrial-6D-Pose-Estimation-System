import numpy as np

class RandomBrightness:
    """Multiply image by random factor between 0.7-1.3"""
    def __call__(self, image):
        factor = np.random.uniform(0.7, 1.3)
        return np.clip(image * factor, 0, 255).astype(np.uint8)

class ToTensor:
    """Convert H x W x C image to C x H x W tensor"""
    def __call__(self, image):
        return image.transpose(2,0,1) / 255.0


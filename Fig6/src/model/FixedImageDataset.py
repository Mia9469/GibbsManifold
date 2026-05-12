import torch
import numpy as np
import os
from torch.utils.data import Dataset

class FixedNatPatchDataset(Dataset):
    """Dataset of natural image patches"""
    def __init__(self, N, width, height, border=0, fpath=None):
        """
        Parameters
        ----------
        N: int
            number of patches
        width: int
            width of patches
        height: int
            height of patches
        border: int
            border to avoid when sampling patches
        fpath: str
            path to image file
        """
        self.N = N
        self.width = width
        self.height = height
        self.border = border

        # Default path with IMAGES.mat
        if fpath is None:
            fpath = os.path.join(os.path.dirname(__file__), "IMAGES.mat")

        self.fpath = fpath
        self.images = torch.zeros(N, width, height)
        # initialize patches
        self.extract_patches_()

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.images[idx]

    def extract_patches_(self):
        """Extract patches from the images"""
        import scipy.io as sio
        
        # Load images
        if self.fpath.endswith('.mat'):
            try:
                img_data = sio.loadmat(self.fpath)
                # IMAGES.mat typically contains a variable named 'IMAGES'
                if 'IMAGES' in img_data:
                    images = img_data['IMAGES']
                else:
                    # Try the first variable that looks like an image array
                    for key in img_data:
                        if isinstance(img_data[key], np.ndarray) and len(img_data[key].shape) >= 2:
                            images = img_data[key]
                            break
            except:
                raise ValueError(f"Could not load images from {self.fpath}")
        else:
            # Add support for other image formats if needed
            raise ValueError(f"Unsupported file format: {self.fpath}")
        
        if isinstance(images, np.ndarray):
            # Handle different image formats
            if len(images.shape) == 3:
                # Format is [height, width, num_images]
                pass
            elif len(images.shape) == 4:
                # Format is [num_images, height, width, channels]
                # Convert to [height, width, num_images]
                images = np.transpose(images[:, :, :, 0], (1, 2, 0))
        
        # Get dimensions
        img_height, img_width, num_images = images.shape
        
        # Ensure borders don't make patches too small
        if img_height - 2*self.border < self.height or img_width - 2*self.border < self.width:
            self.border = 0
            print("Warning: Border too large for image size, setting border to 0")
        
        # Sample patches
        counter = 0
        while counter < self.N:
            # Choose a random image
            img_idx = np.random.randint(0, num_images)
            img = images[:, :, img_idx]
            
            # Choose a random location (respecting border)
            x = np.random.randint(self.border, img_height - self.height - self.border)
            y = np.random.randint(self.border, img_width - self.width - self.border)
            
            # Extract patch
            crop = torch.tensor(img[x:x+self.height, y:y+self.width], dtype=torch.float32)
            
            # Normalize patch by subtracting mean
            self.images[counter] = crop - crop.mean()
            counter += 1

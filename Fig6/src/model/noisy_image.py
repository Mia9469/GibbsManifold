import numpy as np
from skimage import util

def add_gaussian_noise(image, mean=0, var=0.01):
    """
    Add Gaussian noise to an image.

    Parameters:
    - image: Input image (numpy array).
    - mean: Mean of the Gaussian noise.
    - var: Variance of the Gaussian noise.

    Returns:
    - Noisy image.
    """
    noisy_image = util.random_noise(image, mode='gaussian', mean=mean, var=var)
    return noisy_image
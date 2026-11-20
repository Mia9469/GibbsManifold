import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import argparse
import time
import matplotlib.pyplot as plt
import scipy.io as scio
from torch.utils.tensorboard import SummaryWriter
from matplotlib.backends.backend_pdf import PdfPages
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import seaborn as sns
from scipy import stats

# Absolute import since parent directory is a namespace package
from src.model.SparseNetNew import SparseNet
from basis_analysis import linear_independence, orthogonality, mutual_information, decorrelation

def add_gaussian_noise(images, noise_level):
    """Add Gaussian noise to images"""
    return images + noise_level * torch.randn_like(images)

def add_salt_pepper_noise(images, noise_density):
    """Add Salt and Pepper noise to images"""
    noisy_images = images.clone()
    noise_mask = torch.rand_like(images) < noise_density
    salt_mask = torch.rand_like(images) < 0.5
    pepper_mask = ~salt_mask
    
    # Set salt (1) and pepper (0) values
    noisy_images[noise_mask & salt_mask] = 1.0
    noisy_images[noise_mask & pepper_mask] = 0.0
    
    return noisy_images

def add_speckle_noise(images, noise_level):
    """Add Speckle noise (multiplicative noise) to images"""
    noise = noise_level * torch.randn_like(images)
    return images * (1 + noise)

def visualize_dictionary(model_path, output_dir):
    """
    Visualize the dictionary atoms (filters) as a grid of images.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model (.pth file)
    output_dir : str
        Directory to save the visualization
    """
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    
    # Extract dictionary
    model_dict = model.U.weight.data.clone()
    
    n_atoms = model_dict.shape[0]
    patch_size = int(np.sqrt(model_dict.shape[1]))
    
    # Create a grid layout
    grid_size = int(np.ceil(np.sqrt(n_atoms)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    
    # Normalize each filter for better visualization
    for i in range(n_atoms):
        row, col = i // grid_size, i % grid_size
        atom = model_dict[i].cpu().numpy().reshape(patch_size, patch_size)
        
        # Normalize atom for visualization
        atom = (atom - atom.min()) / (atom.max() - atom.min() + 1e-8)
        
        if grid_size > 1:
            axes[row, col].imshow(atom, cmap='gray')
            axes[row, col].axis('off')
        else:
            axes[col].imshow(atom, cmap='gray')
            axes[col].axis('off')
    
    # Hide any empty subplots
    for i in range(n_atoms, grid_size * grid_size):
        row, col = i // grid_size, i % grid_size
        if grid_size > 1:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save as both PNG and PDF for high-quality visualization
    base_name = os.path.basename(model_path).replace('.pth', '')
    plt.savefig(os.path.join(output_dir, f'{base_name}_dictionary.png'))
    plt.savefig(os.path.join(output_dir, f'{base_name}_dictionary.pdf'))
    plt.close()
    
    print(f"Dictionary visualization saved to {output_dir}")
    
    return model_dict

def plot_all_metrics(results, noise_type, constraint, output_dir):
    """Plot all metrics vs noise level"""
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Get the noise levels and sort them
    noise_levels = sorted(list(results.keys()))
    
    # Define the metrics to plot
    metrics = ['loss', 'independence', 'orthogonality', 'linear_independence', 'ssim', 'CV', 'ICE', 'cost', 'cvl']
    titles = ['Mean Squared Error', 'Independence (%)', 'Orthogonality', 'Linear Independence', 
              'Structural Similarity', 'Coefficient of Variation', 'Information Content Entropy', 
              'Mean Coefficient Value', 'CV per Component']
    
    # Create a 3x3 grid of subplots 
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # Plot each metric
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        if i < len(axes):
            metric_values = [results[level][metric] for level in noise_levels]
            
            # Plot the values
            axes[i].plot(noise_levels, metric_values, marker='o', linewidth=2)
            axes[i].set_title(title)
            axes[i].set_xlabel('Noise Level')
            axes[i].set_ylabel(metric)
            axes[i].grid(True)
    
    plt.tight_layout()
    # Save as PNG and PDF
    plt.savefig(os.path.join(output_dir, f'{noise_type}_all_metrics.png'))
    plt.savefig(os.path.join(output_dir, f'{noise_type}_all_metrics.pdf'))
    plt.close()
    
    # Also save individual metrics as separate PDF files for better quality
    for metric, title in zip(metrics, titles):
        plt.figure(figsize=(10, 6))
        metric_values = [results[level][metric] for level in noise_levels]
        plt.plot(noise_levels, metric_values, marker='o', linewidth=2)
        plt.title(title)
        plt.xlabel('Noise Level')
        plt.ylabel(metric)
        plt.grid(True)
        plt.tight_layout()
        # Save individual plot as PDF
        plt.savefig(os.path.join(output_dir, f'{noise_type}_{metric}.pdf'))
        plt.close()

def test_model_with_noise(model_path, noise_type, noise_levels, output_dir=None, constraint='s'):
    """
    Test model performance under different noise levels
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model (.pth file)
    noise_type : str
        Type of noise ('gaussian', 'salt_pepper', or 'speckle')
    noise_levels : list
        List of noise levels/densities to test
    output_dir : str, optional
        Directory to save results
    constraint : str
        Constraint type to use ('s', 'cv', or 'null')
    """
    # Create specific output directory for this constraint
    if output_dir:
        constraint_dir = os.path.join(output_dir, f"constraint_{constraint}")
        if not os.path.exists(constraint_dir):
            os.makedirs(constraint_dir)
    else:
        constraint_dir = None

    # Set up tensorboard if output_dir is provided
    writer = None
    if constraint_dir:
        writer = SummaryWriter(os.path.join(constraint_dir, f'noise_test_{noise_type}'))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    print(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location=device)
    
    # Extract the dictionary (weight matrix) from the model
    model_dict = model.U.weight.data.clone()
    print(f"Dictionary shape: {model_dict.shape}")
    
    # Extract patch size from model
    patch_size = int(np.sqrt(model_dict.shape[1]))
    print(f"Using {patch_size}x{patch_size} patches")
    
    # Generate synthetic test data
    print("Generating synthetic test data...")
    num_samples = 100
    input_dim = patch_size * patch_size
    
    # Create random patches with values between 0 and 1
    synthetic_data = torch.rand(num_samples, input_dim).to(device)
    
    # Also include a 0.0 noise level for baseline comparison
    all_noise_levels = [0.0] + noise_levels
    
    results = {}
    
    # Loop through noise levels
    for noise_level in all_noise_levels:
        noise_type_label = "baseline" if noise_level == 0.0 else noise_type
        print(f"Testing with {noise_type_label} noise at level {noise_level}")
        
        # Initialize metrics accumulators
        running_metrics = {
            'loss': 0.0,
            'independence': 0.0,
            'orthogonality': 0.0,
            'linear_independence': 0.0,
            'ssim': 0.0,
            'CV': 0.0,
            'ICE': 0.0,
            'cost': 0.0,
            'cvl': 0.0
        }
        c = 0
        
        # Process in smaller batches
        batch_size = 32
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
        
        for i in range(num_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            img_batch = synthetic_data[start_idx:end_idx]
            
            # Add noise (but not for baseline)
            if noise_level > 0:
                if noise_type == 'gaussian':
                    noisy_img_batch = add_gaussian_noise(img_batch, noise_level)
                elif noise_type == 'salt_pepper':
                    noisy_img_batch = add_salt_pepper_noise(img_batch, noise_level)
                elif noise_type == 'speckle':
                    noisy_img_batch = add_speckle_noise(img_batch, noise_level)
                else:
                    raise ValueError(f"Unknown noise type: {noise_type}")
            else:
                # No noise for baseline
                noisy_img_batch = img_batch.clone()
            
            # Use the dictionary for sparse coding and reconstruction
            with torch.no_grad():
                # Forward pass with the trained model
                sparse_coef = model.encode(noisy_img_batch)
                pred = model.decode(sparse_coef)
                
                # Calculate MSE loss
                loss = ((img_batch - pred) ** 2).mean().item()
                
                # ====== Calculate dictionary metrics ======
                # Convert to numpy for metric calculations
                U = model_dict.cpu().numpy()
                
                # Calculate independence/decorrelation
                independence = np.sum(decorrelation(U)) / (U.shape[0] * (U.shape[0] - 1)) * 100
                
                # Calculate orthogonality
                ortho_values = []
                for i1 in range(U.shape[0]):
                    for j1 in range(U.shape[0]):
                        if i1 != j1:
                            ortho_values.append(orthogonality(U[i1], U[j1]))
                orthogonality_metric = np.mean(ortho_values)
                
                # Linear independence
                lin_independence = linear_independence(U)
                
                # ====== Calculate coefficient metrics ======
                # Convert sparse coefficients to numpy
                w = sparse_coef.cpu().numpy()
                
                # Calculate CV of coefficients (Coefficient of Variation)
                mean_coef = np.mean(np.abs(w))
                std_coef = np.std(np.abs(w))
                cv = std_coef / (mean_coef + 1e-8)  # Add epsilon to prevent division by zero
                
                # Calculate CVL - CV per component
                cvl = 0
                num_components = 0
                for v in range(w.shape[1]):
                    mean_abs = np.mean(np.abs(w[:, v]))
                    if mean_abs > 1e-8:  # Only count components with non-zero activity
                        cvl += np.std(w[:, v]) / (mean_abs + 1e-8)
                        num_components += 1
                cvl = cvl / (num_components + 1e-8)  # Normalize by active components
                
                # Calculate ICE (Information Content Entropy)
                ent = 0
                for v in range(w.shape[1]):
                    coef_abs = np.abs(w[:, v])
                    if np.mean(coef_abs) > 1e-8:  # Only analyze active components
                        hist, bin_edges = np.histogram(coef_abs, bins=30, range=(0, np.max(coef_abs) + 1e-8))
                        probabilities = hist / (np.sum(hist) + 1e-8)
                        prob = probabilities[probabilities > 0]
                        if len(prob) > 0:
                            entropy = -np.sum(prob * np.log2(prob + 1e-8))
                            weight_factor = np.mean(coef_abs) / (np.std(coef_abs) + 1e-8)
                            ent = max(ent, entropy * weight_factor)
                
                # Calculate cost (using L1 norm as true sparsity measure)
                cost = np.mean(np.abs(w))  # L1 norm (true sparsity measure)
                
                # Calculate SSIM (Structural Similarity Index) between original and reconstructed images
                ssim_values = []
                for k in range(min(5, img_batch.size(0))):
                    img_np = img_batch[k].reshape(patch_size, patch_size).cpu().numpy()
                    pred_np = pred[k].reshape(patch_size, patch_size).cpu().numpy()
                    ssim_values.append(ssim(img_np, pred_np, data_range=1.0))
                ssim_metric = np.mean(ssim_values)
            
            # Accumulate metrics
            running_metrics['loss'] += loss
            running_metrics['independence'] += independence
            running_metrics['orthogonality'] += orthogonality_metric
            running_metrics['linear_independence'] += lin_independence
            running_metrics['ssim'] += ssim_metric
            running_metrics['CV'] += cv
            running_metrics['ICE'] += ent
            running_metrics['cost'] += cost
            running_metrics['cvl'] += cvl
            c += 1
            
            # Save example images for visualization
            if i == 0 and constraint_dir:
                fig, axes = plt.subplots(3, 3, figsize=(12, 12))
                for idx in range(min(3, img_batch.shape[0])):
                    # Original image
                    axes[idx, 0].imshow(img_batch[idx].reshape(patch_size, patch_size).cpu().numpy(), cmap='gray')
                    axes[idx, 0].set_title('Original')
                    
                    # Noisy image
                    axes[idx, 1].imshow(noisy_img_batch[idx].reshape(patch_size, patch_size).cpu().numpy(), cmap='gray')
                    axes[idx, 1].set_title(f'Noisy ({noise_type_label})')
                    
                    # Reconstructed image
                    axes[idx, 2].imshow(pred[idx].reshape(patch_size, patch_size).cpu().detach().numpy(), cmap='gray')
                    axes[idx, 2].set_title(f'Reconstructed')
                
                plt.tight_layout()
                plt.savefig(os.path.join(constraint_dir, f'{noise_type}_{noise_level}_examples.png'))
                plt.savefig(os.path.join(constraint_dir, f'{noise_type}_{noise_level}_examples.pdf'))
                plt.close()
        
        # Calculate average metrics
        avg_metrics = {k: v/c if c > 0 else 0 for k, v in running_metrics.items()}
        
        # Store results
        results[noise_level] = avg_metrics
        
        # Write to tensorboard if available
        if writer:
            for metric, value in avg_metrics.items():
                writer.add_scalar(f'{noise_type}_{metric}', value, int(noise_level * 100))
            
        print(f"Results for {noise_type_label} noise level {noise_level}:")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Plot results and save metrics
    if constraint_dir:
        plot_all_metrics(results, noise_type, constraint, constraint_dir)
        
        # Save results to matlab file
        output_dict = {"noise_levels": list(results.keys())}
        for metric in results[list(results.keys())[0]].keys():
            output_dict[metric] = [results[level][metric] for level in results.keys()]
        
        scio.savemat(
            os.path.join(constraint_dir, f'{noise_type}_noise_results.mat'),
            output_dict
        )
    
    return results

def compare_model_results(base_model_paths, noise_model_paths, noise_levels, noise_type, output_dir=None):
    """
    Compare performance of base models vs models trained with noise
    
    Parameters:
    -----------
    base_model_paths : dict
        Dictionary of base model paths for each constraint type
    noise_model_paths : dict
        Dictionary of noise-trained model paths for each constraint and noise level
    noise_levels : list
        List of noise levels for comparison
    noise_type : str
        Type of noise ('gaussian', 'salt_pepper', or 'speckle')
    output_dir : str, optional
        Directory to save comparison results
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    constraints = list(base_model_paths.keys())
    
    # Store all results
    all_results = {}
    
    # For each constraint, test both model types
    for constraint in constraints:
        base_model_path = base_model_paths[constraint]
        
        print(f"\nComparing models with constraint '{constraint}'")
        print(f"Base model: {base_model_path}")
        print(f"Noise-trained models: {len(noise_model_paths[constraint])} models")
        
        # Dictionary to store constraint-specific results
        constraint_results = {
            'base': {},
            'noise_trained': {}
        }
        
        # Test base model with noise
        print("\nTesting base model with noise...")
        base_results = test_model_with_noise(
            base_model_path,
            noise_type,
            noise_levels,
            output_dir=os.path.join(output_dir, 'base_model_results') if output_dir else None,
            constraint=constraint
        )
        constraint_results['base'] = base_results
        
        # Test noise-trained models at their corresponding noise levels
        print("\nTesting noise-trained models...")
        noise_results = {}
        
        for noise_level in noise_levels:
            if noise_level in noise_model_paths[constraint]:
                noise_model_path = noise_model_paths[constraint][noise_level]
                print(f"Testing noise-trained model with {noise_type} noise level {noise_level}")
                
                # Test this model with this specific noise level
                level_results = test_model_with_noise(
                    noise_model_path,
                    noise_type,
                    [noise_level],  # Only test at the trained noise level
                    output_dir=os.path.join(output_dir, 'noise_model_results') if output_dir else None,
                    constraint=constraint
                )
                
                # Store this result
                if noise_level in level_results:
                    noise_results[noise_level] = level_results[noise_level]
        
        constraint_results['noise_trained'] = noise_results
        all_results[constraint] = constraint_results
        
        # Generate comparison plots for this constraint
        if output_dir:
            comparison_dir = os.path.join(output_dir, 'comparisons', constraint)
            if not os.path.exists(comparison_dir):
                os.makedirs(comparison_dir)
            
            # Create comparison plots for key metrics
            metrics = ['loss', 'independence', 'ssim', 'CV', 'ICE', 'cost']
            metric_names = ['MSE Loss', 'Independence (%)', 'SSIM', 'Coefficient Variation', 
                           'Information Content', 'Mean Coefficient']
            
            for metric, metric_name in zip(metrics, metric_names):
                plt.figure(figsize=(10, 6))
                
                # Get common noise levels between both result sets
                common_levels = sorted([
                    level for level in noise_levels 
                    if level in base_results and level in noise_results
                ])
                
                if common_levels:
                    # Base model performance at different noise levels
                    base_values = [base_results[level][metric] for level in common_levels]
                    
                    # Noise-trained model performance at corresponding noise levels
                    noise_values = [noise_results[level][metric] for level in common_levels]
                    
                    # Plot both curves
                    plt.plot(common_levels, base_values, marker='o', label='Base Model', linewidth=2)
                    plt.plot(common_levels, noise_values, marker='s', label='Noise-Trained Model', linewidth=2)
                    
                    plt.title(f'{metric_name} vs {noise_type.title()} Noise Level (Constraint: {constraint})')
                    plt.xlabel('Noise Level')
                    plt.ylabel(metric_name)
                    plt.grid(True)
                    plt.legend()
                    
                    # Save as both PNG and PDF
                    plt.savefig(os.path.join(comparison_dir, f'{noise_type}_{metric}_comparison.png'))
                    plt.savefig(os.path.join(comparison_dir, f'{noise_type}_{metric}_comparison.pdf'))
                    plt.close()
    
    return all_results

def modify_sparsenet_class():
    """Monkey patch SparseNet class to add encode/decode methods for analysis"""
    from src.model.SparseNetNew import SparseNet
    
    # Add encode and decode methods for easier evaluation
    def encode(self, x):
        """Encode input using current dictionary"""
        with torch.no_grad():
            # First fit to get sparse coefficients
            self.ista_(x)
            # Return the resulting coefficients
            return self.R.clone()
    
    def decode(self, coefficients):
        """Decode coefficients using current dictionary"""
        with torch.no_grad():
            return self.U(coefficients)
    
    # Monkey patch the class
    SparseNet.encode = encode
    SparseNet.decode = decode
    
    print("SparseNet class has been modified to support encoding/decoding for analysis")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Analyze SparseNet Models')
    parser.add_argument('--analysis_type', type=str, required=True, 
                        choices=['visualize', 'test_noise', 'compare'],
                        help='Type of analysis to perform')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the model file (.pth) for visualization or testing')
    parser.add_argument('--base_model_dir', type=str, default=None,
                        help='Directory containing base models for comparison')
    parser.add_argument('--noise_model_dir', type=str, default=None,
                        help='Directory containing noise-trained models for comparison')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                        help='Directory to save analysis results')
    parser.add_argument('--noise_type', type=str, default='gaussian',
                        choices=['gaussian', 'salt_pepper', 'speckle'],
                        help='Type of noise to test with')
    parser.add_argument('--constraint', type=str, default='s',
                        choices=['s', 'cv', 'null'],
                        help='Constraint type of the model')
    
    args = parser.parse_args()
    
    # Apply monkey patch to SparseNet class for analysis
    modify_sparsenet_class()
    
    if args.analysis_type == 'visualize':
        if not args.model_path:
            print("Error: model_path is required for visualization")
            sys.exit(1)
            
        print(f"Visualizing dictionary atoms from {args.model_path}")
        visualize_dictionary(args.model_path, args.output_dir)
        
    elif args.analysis_type == 'test_noise':
        if not args.model_path:
            print("Error: model_path is required for noise testing")
            sys.exit(1)
            
        # Define noise levels based on noise type
        if args.noise_type == 'gaussian':
            noise_levels = [0.05, 0.1, 0.2, 0.3, 0.5]
        elif args.noise_type == 'salt_pepper':
            noise_levels = [0.05, 0.1, 0.2, 0.3]
        elif args.noise_type == 'speckle':
            noise_levels = [0.05, 0.1, 0.2, 0.3, 0.5]
            
        print(f"Testing model with {args.noise_type} noise")
        results = test_model_with_noise(
            args.model_path,
            args.noise_type,
            noise_levels,
            output_dir=args.output_dir,
            constraint=args.constraint
        )
        print("Noise testing completed and results saved")
        
    elif args.analysis_type == 'compare':
        if not args.base_model_dir or not args.noise_model_dir:
            print("Error: both base_model_dir and noise_model_dir are required for comparison")
            sys.exit(1)
            
        # Define noise levels
        if args.noise_type == 'gaussian':
            noise_levels = [0.05, 0.1, 0.2, 0.3, 0.5]
        elif args.noise_type == 'salt_pepper':
            noise_levels = [0.05, 0.1, 0.2, 0.3]
        elif args.noise_type == 'speckle':
            noise_levels = [0.05, 0.1, 0.2, 0.3, 0.5]
            
        # Find base models for different constraints
        constraints = ['s', 'cv', 'null']
        base_model_paths = {}
        for constraint in constraints:
            base_model_path = os.path.join(args.base_model_dir, f'sparsenet_{constraint}_final.pth')
            if os.path.exists(base_model_path):
                base_model_paths[constraint] = base_model_path
                
        if not base_model_paths:
            print("Error: No base models found in the specified directory")
            sys.exit(1)
            
        # Find noise-trained models for different constraints and noise levels
        noise_model_paths = {constraint: {} for constraint in base_model_paths.keys()}
        
        for constraint in base_model_paths.keys():
            constraint_dir = os.path.join(args.noise_model_dir, f'constraint_{constraint}_noise')
            if os.path.exists(constraint_dir):
                for noise_level in noise_levels:
                    model_filename = f'model_{constraint}_{args.noise_type}_{noise_level}.pth'
                    model_path = os.path.join(constraint_dir, model_filename)
                    if os.path.exists(model_path):
                        noise_model_paths[constraint][noise_level] = model_path
                        
        # Verify we have at least some noise models to compare
        has_models = False
        for constraint in noise_model_paths:
            if noise_model_paths[constraint]:
                has_models = True
                break
                
        if not has_models:
            print("Error: No noise-trained models found in the specified directory")
            sys.exit(1)
            
        print("Comparing base models with noise-trained models")
        compare_model_results(
            base_model_paths,
            noise_model_paths,
            noise_levels,
            args.noise_type,
            output_dir=args.output_dir
        )
        print("Comparison completed and results saved")

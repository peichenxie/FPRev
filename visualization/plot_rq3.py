"""
Visualization script for RQ3 Performance Evaluation Results

This script generates the CPU vs GPU performance comparison from the FPRev paper,
showing the runtime comparison of BasicFPRev and FPRev algorithms across different
hardware platforms.

Research Question 3 (RQ3):
How does FPRev perform on different hardware architectures (CPU vs GPU)?

The visualization creates two subplots:
- Left: CPU performance (with detected CPU model)
- Right: GPU performance (with detected GPU model)

Each subplot shows:
- Log-scale both axes representing runtime vs. problem size
- Green line for BasicFPRev algorithm  
- Blue line for FPRev algorithm

Usage:
------
$ python visualization/plot_rq3.py

Output:
-------
- Displays interactive matplotlib figure
- Saves plot to outputs/rq3_performance.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import platform
import subprocess
import re


def detect_cpu_info():
    """Detect CPU information for subplot title."""
    try:
        if platform.system() == "Linux":
            # Try to get CPU info from /proc/cpuinfo
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            
            # Look for model name
            model_match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
            if model_match:
                cpu_name = model_match.group(1).strip()
                # Simplify the name by removing extra details
                cpu_name = re.sub(r"\s+@.*", "", cpu_name)  # Remove frequency
                cpu_name = re.sub(r"\s+CPU", "", cpu_name)  # Remove redundant CPU
                return cpu_name
            
            # Fallback: try lscpu
            result = subprocess.run(["lscpu"], capture_output=True, text=True)
            if result.returncode == 0:
                model_match = re.search(r"Model name:\s*(.+)", result.stdout)
                if model_match:
                    return model_match.group(1).strip()
                    
    except Exception:
        pass
    
    return "CPU"


def detect_gpu_info():
    """Detect GPU information for subplot title."""
    try:
        # Try nvidia-smi first
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"], 
                              capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split('\n')[0]
            return gpu_name
        
        # Try lspci for other GPUs
        result = subprocess.run(["lspci"], capture_output=True, text=True)
        if result.returncode == 0:
            gpu_matches = re.findall(r"VGA.*?:\s*(.+)", result.stdout)
            if gpu_matches:
                gpu_name = gpu_matches[0]
                # Clean up the name
                gpu_name = re.sub(r"\[.*?\]", "", gpu_name).strip()
                return gpu_name
                
    except Exception:
        pass
    
    return "GPU"


def get_hardware_titles():
    """Get hardware-specific titles for the subplots."""
    cpu_name = detect_cpu_info()
    gpu_name = detect_gpu_info()
    
    # Try to detect core count for CPU
    try:
        import multiprocessing
        cpu_cores = multiprocessing.cpu_count()
        cpu_title = f"{cpu_name} ({cpu_cores} cores)"
    except:
        cpu_title = cpu_name
    
    return cpu_title, gpu_name


def load_rq3_data(filepath="outputs/rq3.csv"):
    """Load and parse the RQ3 CSV data with proper multi-index columns."""
    # Read with multi-level headers
    df = pd.read_csv(filepath, header=[0, 1], index_col=0)
    
    # Convert index to numeric (number of summands)
    df.index = pd.to_numeric(df.index)
    
    return df


def plot_rq3_results(df):
    """Generate the performance comparison plots for RQ3."""
    
    # Get hardware-specific titles
    cpu_title, gpu_title = get_hardware_titles()
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    platforms = ["CPU", "GPU"]
    platform_titles = [cpu_title, gpu_title]
    algorithms = ["BasicFPRev", "FPRev"]
    colors = ['green', 'blue']
    markers = ['s', '^']
    
    for i, (platform, title) in enumerate(zip(platforms, platform_titles)):
        ax = axes[i]
        
        # Plot each algorithm
        for j, algorithm in enumerate(algorithms):
            if (platform, algorithm) in df.columns:
                # Filter out NaN values
                data = df[(platform, algorithm)].dropna()
                if not data.empty:
                    ax.loglog(data.index, data.values, 
                            color=colors[j], marker=markers[j], 
                            label=algorithm, linewidth=2, markersize=6)
        
        # Customize subplot
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of summands (n)', fontsize=10)
        if i == 0:  # Only label y-axis on leftmost plot
            ax.set_ylabel('Runtime (seconds)', fontsize=10)
        
        # Set axis limits for better visualization
        ax.set_ylim(0.0001, 10)
        ax.set_xlim(3, 2000)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(fontsize=9, loc='upper left')
        
        # Set specific y-tick labels
        ax.set_yticks([0.0001, 0.001, 0.01, 0.1, 1, 10])
        ax.set_yticklabels(['0.0001', '0.001', '0.01', '0.1', '1', '10'])
        
        # Set x-tick labels
        ax.set_xticks([4, 16, 64, 256, 1024])
        ax.set_xticklabels(['4', '16', '64', '256', '1024'])
    
    plt.tight_layout()
    
    return fig


def main():
    """Main function to load data and generate visualization."""
    try:
        # Load the data
        print("Loading RQ3 data from outputs/rq3.csv...")
        df = load_rq3_data()
        print(f"Data loaded successfully. Shape: {df.shape}")
        print("\nData summary:")
        print(df.head())
        
        # Detect hardware info
        cpu_title, gpu_title = get_hardware_titles()
        print(f"\nDetected hardware:")
        print(f"CPU: {cpu_title}")
        print(f"GPU: {gpu_title}")
        
        # Generate the plot
        print("\nGenerating performance comparison plots...")
        fig = plot_rq3_results(df)
        
        # Save the figure
        os.makedirs("outputs", exist_ok=True)
        output_path = "outputs/rq3_performance.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        
        # Show the plot
        plt.show()
        
    except FileNotFoundError:
        print("Error: outputs/rq3.csv not found. Please run experiments/rq3.py first.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 
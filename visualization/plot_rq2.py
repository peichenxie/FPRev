"""
Visualization script for RQ2 Performance Evaluation Results

This script generates Figure 6 from the FPRev paper, showing the runtime comparison
of BasicFPRev and FPRev algorithms across different mathematical operations in NumPy.

Research Question 2 (RQ2):
How does FPRev perform on different types of numerical operations beyond simple summation?

The visualization creates three subplots for different operations:
- Dot product: Vector dot product operations
- Matrix-vector multiplication (GEMV): General matrix-vector multiplication  
- Matrix multiplication (GEMM): General matrix-matrix multiplication

Each subplot shows:
- Log-scale both axes representing runtime vs. problem size
- Green line for BasicFPRev algorithm
- Blue line for FPRev algorithm

Usage:
------
$ python visualization/plot_rq2.py

Output:
-------
- Displays interactive matplotlib figure
- Saves plot to outputs/rq2_performance.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def load_rq2_data(filepath="outputs/rq2.csv"):
    """Load and parse the RQ2 CSV data with proper multi-index columns."""
    # Read with multi-level headers
    df = pd.read_csv(filepath, header=[0, 1], index_col=0)
    
    # Convert index to numeric (number of summands)
    df.index = pd.to_numeric(df.index)
    
    return df


def plot_rq2_results(df):
    """Generate the performance comparison plots for RQ2."""
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    operations = ["Dot", "GEMV", "GEMM"]
    operation_titles = ["Dot product", "Matrix-vector multiplication", "Matrix multiplication"]
    algorithms = ["BasicFPRev", "FPRev"]
    colors = ['green', 'blue']
    markers = ['s', '^']
    
    for i, (operation, title) in enumerate(zip(operations, operation_titles)):
        ax = axes[i]
        
        # Plot each algorithm
        for j, algorithm in enumerate(algorithms):
            if (operation, algorithm) in df.columns:
                # Filter out NaN values
                data = df[(operation, algorithm)].dropna()
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
        ax.set_ylim(0.0001, 100)
        ax.set_xlim(3, 50000)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(fontsize=9, loc='upper left')
        
        # Set specific y-tick labels
        ax.set_yticks([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])
        ax.set_yticklabels(['0.0001', '0.001', '0.01', '0.1', '1', '10', '100'])
        
        # Set x-tick labels based on operation type
        if operation == "Dot":
            ax.set_xticks([4, 16, 64, 256, 1024, 4096, 16384, 32768])
            ax.set_xticklabels(['4', '16', '64', '256', '1024', '4096', '16384', '32768'])
        elif operation == "GEMV":
            ax.set_xticks([4, 16, 64, 256, 1024, 4096])
            ax.set_xticklabels(['4', '16', '64', '256', '1024', '4096'])
        else:  # GEMM
            ax.set_xticks([4, 8, 16, 32, 64, 128, 256, 512])
            ax.set_xticklabels(['4', '8', '16', '32', '64', '128', '256', '512'])
    
    # Add main title and caption
    fig.suptitle('Figure 6: Run time of applying BasicFPRev and FPRev to the dot product, matrix-vector multiplication, and matrix multiplication functions in NumPy', 
                 fontsize=14, y=0.02)
    
    caption = ("The vertical axis represents run time in seconds. "
              "The horizontal axis represents the number of summands n.")
    fig.text(0.5, -0.05, caption, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    
    return fig


def main():
    """Main function to load data and generate visualization."""
    try:
        # Load the data
        print("Loading RQ2 data from outputs/rq2.csv...")
        df = load_rq2_data()
        print(f"Data loaded successfully. Shape: {df.shape}")
        print("\nData summary:")
        print(df.head())
        
        # Generate the plot
        print("\nGenerating performance comparison plots...")
        fig = plot_rq2_results(df)
        
        # Save the figure
        os.makedirs("outputs", exist_ok=True)
        output_path = "outputs/rq2_performance.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        
        # Show the plot
        plt.show()
        
    except FileNotFoundError:
        print("Error: outputs/rq2.csv not found. Please run experiments/rq2.py first.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
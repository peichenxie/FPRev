"""
Visualization script for RQ1 Performance Evaluation Results

This script generates Figure 5 from the FPRev paper, showing the runtime comparison
of NaiveSol, BasicFPRev, and FPRev algorithms across NumPy, PyTorch, and JAX libraries.

The visualization creates three subplots (one for each library) with:
- Log-scale y-axis representing runtime in seconds
- Linear x-axis representing the number of summands (n)
- Three colored lines for each algorithm (NaiveSol=red, BasicFPRev=green, FPRev=blue)

Usage:
------
$ python visualization/plot_rq1.py

Output:
-------
- Displays interactive matplotlib figure
- Optionally saves to outputs/rq1_performance.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def load_rq1_data(filepath="outputs/rq1.csv"):
    """Load and parse the RQ1 CSV data with proper multi-index columns."""
    # Read with multi-level headers
    df = pd.read_csv(filepath, header=[0, 1], index_col=0)
    
    # Convert index to numeric (number of summands)
    df.index = pd.to_numeric(df.index)
    
    return df


def plot_rq1_results(df):
    """Generate the performance comparison plots for RQ1."""
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Figure 5: Run time of applying NaiveSol, BasicFPRev, and FPRev to the summation functions in NumPy, PyTorch, and JAX', 
                 fontsize=14, y=0.02)
    
    libraries = ["NumPy", "PyTorch", "JAX"]
    algorithms = ["NaiveSol", "BasicFPRev", "FPRev"]
    colors = ['red', 'green', 'blue']
    markers = ['o', 's', '^']
    
    for i, library in enumerate(libraries):
        ax = axes[i]
        
        # Plot each algorithm
        for j, algorithm in enumerate(algorithms):
            if (library, algorithm) in df.columns:
                # Filter out NaN values
                data = df[(library, algorithm)].dropna()
                if not data.empty:
                    ax.loglog(data.index, data.values, 
                            color=colors[j], marker=markers[j], 
                            label=algorithm, linewidth=2, markersize=6)
        
        # Customize subplot
        ax.set_title(library, fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of summands (n)', fontsize=10)
        if i == 0:  # Only label y-axis on leftmost plot
            ax.set_ylabel('Runtime (seconds)', fontsize=10)
        
        # Set y-axis limits and ticks for better comparison
        ax.set_ylim(0.0001, 10)
        ax.set_xlim(3, 20000)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(fontsize=9, loc='upper left')
        
        # Set specific y-tick labels
        ax.set_yticks([0.0001, 0.001, 0.01, 0.1, 1, 10])
        ax.set_yticklabels(['0.0001', '0.001', '0.01', '0.1', '1', '10'])
        
        # Set specific x-tick labels
        ax.set_xticks([4, 16, 64, 256, 1024, 4096, 16384])
        ax.set_xticklabels(['4', '16', '64', '256', '1024', '4096', '16384'])
    
    # Add caption
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
        print("Loading RQ1 data from outputs/rq1.csv...")
        df = load_rq1_data()
        print(f"Data loaded successfully. Shape: {df.shape}")
        print("\nData summary:")
        print(df.head())
        
        # Generate the plot
        print("\nGenerating performance comparison plots...")
        fig = plot_rq1_results(df)
        
        # Save the figure
        os.makedirs("outputs", exist_ok=True)
        output_path = "outputs/rq1_performance.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        
        # Show the plot
        plt.show()
        
    except FileNotFoundError:
        print("Error: outputs/rq1.csv not found. Please run experiments/rq1.py first.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

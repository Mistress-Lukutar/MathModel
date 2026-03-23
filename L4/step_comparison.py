"""
Step Comparison for Lab 4.

Compares numerical solutions obtained with different step sizes.

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-19
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional


class StepComparator:
    """
    Comparator for solutions with different step sizes.
    
    Visualizes how the numerical solution converges to the
    analytical solution as step size decreases.
    """
    
    def __init__(self, convergence_results: List[Dict],
                 reference_solution: Optional[Dict] = None,
                 state_names: Optional[List[str]] = None):
        """
        Initialize step comparator.
        
        Args:
            convergence_results: Results from ConvergenceAnalyzer
            reference_solution: Optional reference (analytical) solution
            state_names: Optional list of state names
        """
        self.results = convergence_results
        self.reference = reference_solution
        
        if state_names is None and convergence_results:
            n_states = convergence_results[0]['solution']['y'].shape[0]
            self.state_names = [f"P_{i+1}" for i in range(n_states)]
        else:
            self.state_names = state_names
    
    def plot_error_vs_step(self, output_dir: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 8),
                          suffix: str = "") -> str:
        """
        Plot error vs step size (convergence plot).
        
        Args:
            output_dir: Directory to save figure
            figsize: Figure size
            
        Returns:
            Path to saved figure
        """
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(script_dir)
            output_dir = os.path.join(root_dir, 'Output')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data
        steps = []
        errors = []
        
        for r in self.results:
            if r['max_error'] is not None:
                steps.append(r['h'])
                errors.append(r['max_error'])
        
        if len(steps) < 2:
            print("Not enough data for convergence plot")
            return None
        
        steps = np.array(steps)
        errors = np.array(errors)
        
        # Create plot with 2 subplots: linear with legend and log-log
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Fit line to estimate order (for log-log plot)
        log_h = np.log(steps)
        log_e = np.log(errors)
        A = np.vstack([log_h, np.ones(len(log_h))]).T
        slope, intercept = np.linalg.lstsq(A, log_e, rcond=None)[0]
        
        # Left: Linear scale with legend showing values
        ax1 = axes[0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(steps)))
        for i, (h, e) in enumerate(zip(steps, errors)):
            ax1.plot(h, e, 'o', color=colors[i], markersize=12, 
                    label=f'h={h:.3f}: {e:.2e}')
        ax1.plot(steps, errors, 'b-', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Step Size (h)', fontsize=12)
        ax1.set_ylabel('Maximum Absolute Error', fontsize=12)
        ax1.set_title('Error vs Step Size (Linear)', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10, loc='best', title='Step: Error')
        
        # Right: Log-log scale with fitted line
        ax2 = axes[1]
        ax2.loglog(steps, errors, 'bo-', linewidth=2, markersize=10, label='Numerical')
        
        # Plot fitted line
        h_fit = np.linspace(min(steps), max(steps), 100)
        e_fit = np.exp(intercept) * h_fit**slope
        ax2.loglog(h_fit, e_fit, 'r--', linewidth=2, alpha=0.8,
                  label=f'Fit: O(h^{slope:.2f})')
        
        # Plot theoretical O(h^2) line
        if len(steps) >= 2:
            h_theory = np.array([min(steps), max(steps)])
            scale = errors[-1] / (steps[-1]**2)
            e_theory = scale * h_theory**2
            ax2.loglog(h_theory, e_theory, 'g:', linewidth=2, alpha=0.8,
                      label='Theoretical: O(h²)')
        
        ax2.set_xlabel('Step Size (h)', fontsize=12)
        ax2.set_ylabel('Maximum Absolute Error', fontsize=12)
        ax2.set_title(f'Error vs Step Size (Log-Log), p={slope:.2f}', fontsize=13)
        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend(fontsize=11, loc='best')
        
        title_suffix = f" ({suffix})" if suffix else ""
        plt.suptitle(f'Convergence Analysis - Modified Euler Method{title_suffix}', fontsize=14)
        plt.tight_layout()
        
        filename = f'L4_convergence_{suffix}.png' if suffix else 'L4_convergence.png'
        output_file = os.path.join(output_dir, filename)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved convergence plot: {output_file}")
        return output_file


if __name__ == "__main__":
    # Example usage would require actual solver results
    print("StepComparator module ready for use in L4 workflow")

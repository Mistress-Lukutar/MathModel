"""
Comparison module for L2 vs L1 results.
Compares analytical solution with numerical solution from L1.

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-16
"""

import os
import numpy as np
import matplotlib.pyplot as plt


class L2L1Comparator:
    """
    Compares analytical solution (L2) with numerical solution (L1).
    """
    
    def __init__(self, solver_l2, solution_l1_path=None):
        """
        Initialize comparator.
        
        Args:
            solver_l2 (OperatorSolver): Solved L2 solver instance
            solution_l1_path (str): Path to L1 solution file (.npy)
        """
        self.solver_l2 = solver_l2
        self.solution_l1_path = solution_l1_path
        self.l1_data = None
        self.comparison_results = None
        
        if solution_l1_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(script_dir)
            self.solution_l1_path = os.path.join(root_dir, 'Output', 'L1_solution.npy')
    
    def load_L1_solution(self):
        """Load numerical solution from L1."""
        if not os.path.exists(self.solution_l1_path):
            print(f"Warning: L1 solution not found at {self.solution_l1_path}")
            print("Run L1 first or provide path to solution file.")
            return False
        
        self.l1_data = np.load(self.solution_l1_path, allow_pickle=True).item()
        print(f"Loaded L1 solution: t=[{self.l1_data['t'][0]:.2f}, {self.l1_data['t'][-1]:.2f}], "
              f"{len(self.l1_data['t'])} points")
        return True
    
    def compare(self):
        """
        Perform comparison between L2 and L1 solutions.
        
        Returns:
            dict: Comparison metrics
        """
        if self.l1_data is None and not self.load_L1_solution():
            return None
        
        print("\nComparing L2 (analytical) vs L1 (numerical)...")
        print("=" * 60)
        
        # Evaluate L2 solution at L1 time points
        t_l1 = self.l1_data['t']
        y_l1 = self.l1_data['y']  # shape: (n_states, n_points)
        
        y_l2 = self.solver_l2.evaluate(t_l1)
        
        # Calculate errors
        abs_error = np.abs(y_l2 - y_l1)
        rel_error = np.abs(y_l2 - y_l1) / (y_l1 + 1e-10)
        
        # Per-state metrics
        state_metrics = []
        for i in range(self.solver_l2.n_states):
            metrics = {
                'state': i + 1,
                'max_abs_error': np.max(abs_error[i, :]),
                'mean_abs_error': np.mean(abs_error[i, :]),
                'max_rel_error': np.max(rel_error[i, :]),
                'mean_rel_error': np.mean(rel_error[i, :]),
                'rmse': np.sqrt(np.mean((y_l2[i, :] - y_l1[i, :])**2))
            }
            state_metrics.append(metrics)
        
        # Global metrics
        global_metrics = {
            'max_abs_error': np.max(abs_error),
            'mean_abs_error': np.mean(abs_error),
            'max_rel_error': np.max(rel_error),
            'mean_rel_error': np.mean(rel_error),
            'rmse': np.sqrt(np.mean((y_l2 - y_l1)**2))
        }
        
        self.comparison_results = {
            't': t_l1,
            'y_l1': y_l1,
            'y_l2': y_l2,
            'abs_error': abs_error,
            'rel_error': rel_error,
            'state_metrics': state_metrics,
            'global_metrics': global_metrics
        }
        
        self._print_results()
        return self.comparison_results
    
    def _print_results(self):
        """Print comparison results."""
        print(f"\nGlobal Metrics:")
        print(f"  Max absolute error:  {self.comparison_results['global_metrics']['max_abs_error']:.2e}")
        print(f"  Mean absolute error: {self.comparison_results['global_metrics']['mean_abs_error']:.2e}")
        print(f"  Max relative error:  {self.comparison_results['global_metrics']['max_rel_error']:.2e}")
        print(f"  RMS error:           {self.comparison_results['global_metrics']['rmse']:.2e}")
        
        print(f"\nPer-State Metrics:")
        print(f"{'State':<8}{'Max Abs':<12}{'Mean Abs':<12}{'Max Rel':<12}{'RMSE':<12}")
        print("-" * 60)
        for m in self.comparison_results['state_metrics']:
            print(f"P_{m['state']:<7}{m['max_abs_error']:<12.2e}{m['mean_abs_error']:<12.2e}"
                  f"{m['max_rel_error']:<12.2e}{m['rmse']:<12.2e}")
    
    def plot_comparison(self, output_dir=None):
        """
        Generate comparison plots.
        
        Args:
            output_dir (str): Directory to save plots
        """
        if self.comparison_results is None:
            self.compare()
        
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(script_dir)
            output_dir = os.path.join(root_dir, 'Output')
        
        os.makedirs(output_dir, exist_ok=True)
        
        t = self.comparison_results['t']
        y_l1 = self.comparison_results['y_l1']
        y_l2 = self.comparison_results['y_l2']
        n_states = self.solver_l2.n_states
        
        # Plot 1: All states comparison
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top: L2 vs L1 for all states
        ax = axes[0]
        colors = plt.cm.tab10(np.linspace(0, 1, n_states))
        for i in range(n_states):
            ax.plot(t, y_l2[i, :], '-', color=colors[i], linewidth=2, label=f'P_{i+1} (L2)')
            ax.plot(t, y_l1[i, :], '--', color=colors[i], linewidth=1, alpha=0.7)
        ax.set_xlabel('Time (t)', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('L2 (Analytical) vs L1 (Numerical) - Solid=L2, Dashed=L1', fontsize=14)
        ax.legend(loc='right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        
        # Bottom: Absolute error
        ax = axes[1]
        abs_err = self.comparison_results['abs_error']
        for i in range(n_states):
            ax.semilogy(t, abs_err[i, :] + 1e-15, color=colors[i], label=f'Error P_{i+1}')
        ax.set_xlabel('Time (t)', fontsize=12)
        ax.set_ylabel('Absolute Error (log scale)', fontsize=12)
        ax.set_title(f'Absolute Error - Max: {np.max(abs_err):.2e}', fontsize=14)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'L2_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison plot: {output_file}")
        
        # Plot 2: Individual state comparisons
        n_cols = 3
        n_rows = (n_states + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_states > 1 else [axes]
        
        for i in range(n_states):
            ax = axes[i]
            ax.plot(t, y_l1[i, :], 'b--', linewidth=2, label='L1 (Numerical)')
            ax.plot(t, y_l2[i, :], 'r-', linewidth=1.5, alpha=0.7, label='L2 (Analytical)')
            ax.fill_between(t, y_l1[i, :], y_l2[i, :], alpha=0.2, color='green')
            ax.set_xlabel('Time (t)', fontsize=10)
            ax.set_ylabel('Probability', fontsize=10)
            ax.set_title(f'P_{i+1}(t) - Max Error: {np.max(abs_err[i,:]):.2e}', fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0)
            ax.set_ylim(0, max(np.max(y_l1[i, :]), np.max(y_l2[i, :])) * 1.1)
        
        # Hide extra subplots
        for i in range(n_states, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'L2_individual_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved individual comparison plot: {output_file}")


def compare_solutions(solver_l2, solution_l1_path=None):
    """
    Convenience function to compare L2 and L1 solutions.
    
    Args:
        solver_l2 (OperatorSolver): Solved L2 solver
        solution_l1_path (str): Path to L1 solution
        
    Returns:
        dict: Comparison results
    """
    comparator = L2L1Comparator(solver_l2, solution_l1_path)
    results = comparator.compare()
    if results:
        comparator.plot_comparison()
    return results

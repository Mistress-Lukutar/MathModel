"""
Comparison of L3 numerical solution with L2 analytical solution.

Computes errors, finds maximum deviation intervals, and generates
comparison visualizations.

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-19
"""

import numpy as np
import matplotlib.pyplot as plt
import os


class L3L2Comparator:
    """
    Comparator for L3 numerical solution vs L2 analytical solution.
    """
    
    def __init__(self, l3_solver, l2_solution=None):
        """
        Initialize comparator.
        
        Args:
            l3_solver: ModifiedEulerSolver instance (solved)
            l2_solution: L2 solution data (dict with 't', 'y') or OperatorSolver
        """
        self.l3_solver = l3_solver
        self.n_states = l3_solver.n_states
        
        if l2_solution is not None:
            self.l2_solution = l2_solution
        else:
            self.l2_solution = self._load_l2_solution()
    
    def _load_l2_solution(self):
        """
        Load L2 solution from Output/L2_solution.npy.
        
        Returns:
            dict: L2 solution data
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        l2_file = os.path.join(root_dir, 'Output', 'L2_solution.npy')
        
        if not os.path.exists(l2_file):
            raise FileNotFoundError(
                f"L2 solution not found: {l2_file}\n"
                "Please run Lab 2 first to generate analytical solution."
            )
        
        data = np.load(l2_file, allow_pickle=True).item()
        print(f"Loaded L2 solution: {data['n_states']} states, "
              f"{len(data['t'])} time points")
        return data
    
    def get_l2_at_points(self, t_query):
        """
        Get L2 solution values at query time points.
        
        Args:
            t_query (np.ndarray): Time points
            
        Returns:
            np.ndarray: L2 solution values (n_states x n_points)
        """
        if hasattr(self.l2_solution, 'evaluate'):
            # OperatorSolver instance
            return self.l2_solution.evaluate(t_query)
        
        # Dict with saved solution - interpolate
        y_l2 = np.zeros((self.n_states, len(t_query)))
        for i in range(self.n_states):
            y_l2[i, :] = np.interp(
                t_query,
                self.l2_solution['t'],
                self.l2_solution['y'][i, :]
            )
        return y_l2
    
    def compare(self, t_points=None):
        """
        Perform full comparison at specified time points.
        
        Args:
            t_points (np.ndarray): Time points for comparison.
                If None, uses L3 solver's stored points.
                
        Returns:
            dict: Comparison results
        """
        if t_points is None:
            t_points = self.l3_solver.t_values
        
        # Get solutions
        y_l3 = self.l3_solver.evaluate(t_points)
        y_l2 = self.get_l2_at_points(t_points)
        
        # Compute errors
        abs_error = np.abs(y_l3 - y_l2)
        
        # Relative error (with protection against division by zero)
        rel_error = np.zeros_like(abs_error)
        mask = y_l2 > 1e-10
        rel_error[mask] = abs_error[mask] / y_l2[mask]
        
        # Per-state metrics
        state_metrics = []
        for i in range(self.n_states):
            state_metrics.append({
                'state': i + 1,
                'max_abs_error': np.max(abs_error[i, :]),
                'mean_abs_error': np.mean(abs_error[i, :]),
                'max_rel_error': np.max(rel_error[i, mask[i, :]]) if np.any(mask[i, :]) else 0,
                'mean_rel_error': np.mean(rel_error[i, mask[i, :]]) if np.any(mask[i, :]) else 0,
                'rmse': np.sqrt(np.mean(abs_error[i, :]**2))
            })
        
        # Global metrics
        global_metrics = {
            'max_abs_error': np.max(abs_error),
            'mean_abs_error': np.mean(abs_error),
            'max_rel_error': np.max(rel_error[mask]) if np.any(mask) else 0,
            'mean_rel_error': np.mean(rel_error[mask]) if np.any(mask) else 0,
            'rmse': np.sqrt(np.mean(abs_error**2))
        }
        
        # Find interval of maximum deviation
        max_deviation_idx = np.unravel_index(np.argmax(abs_error), abs_error.shape)
        max_deviation_state = max_deviation_idx[0]
        max_deviation_time_idx = max_deviation_idx[1]
        max_deviation_time = t_points[max_deviation_time_idx]
        
        # Find interval (where error > 0.5 * max for the same state)
        threshold = 0.5 * abs_error[max_deviation_state, max_deviation_time_idx]
        above_threshold = abs_error[max_deviation_state, :] > threshold
        interval_indices = np.where(above_threshold)[0]
        
        if len(interval_indices) > 0:
            max_deviation_interval = (t_points[interval_indices[0]], 
                                     t_points[interval_indices[-1]])
        else:
            max_deviation_interval = (max_deviation_time, max_deviation_time)
        
        return {
            't_points': t_points,
            'y_l3': y_l3,
            'y_l2': y_l2,
            'abs_error': abs_error,
            'rel_error': rel_error,
            'state_metrics': state_metrics,
            'global_metrics': global_metrics,
            'max_deviation': {
                'state': max_deviation_state + 1,
                'time': max_deviation_time,
                'value': abs_error[max_deviation_state, max_deviation_time_idx],
                'relative_value': rel_error[max_deviation_state, max_deviation_time_idx],
                'interval': max_deviation_interval
            },
            'step_size': self.l3_solver.h
        }
    
    def plot_comparison(self, comparison_results, output_file=None):
        """
        Generate comparison plots.
        
        Args:
            comparison_results (dict): Results from compare()
            output_file (str): Path to save figure
            
        Returns:
            matplotlib.figure.Figure: Created figure
        """
        t = comparison_results['t_points']
        y_l3 = comparison_results['y_l3']
        y_l2 = comparison_results['y_l2']
        abs_error = comparison_results['abs_error']
        rel_error = comparison_results['rel_error']
        
        n_states = y_l3.shape[0]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        
        # Main comparison plot (top, spans full width)
        ax_main = plt.subplot(3, 1, 1)
        colors = plt.cm.tab10(np.linspace(0, 1, n_states))
        
        for i in range(n_states):
            # L2 (analytical) - solid line
            ax_main.plot(t, y_l2[i, :], '-', color=colors[i], 
                        linewidth=2, label=f'P_{i+1} L2 (analytical)')
            # L3 (numerical) - dashed line
            ax_main.plot(t, y_l3[i, :], '--', color=colors[i], 
                        linewidth=1.5, alpha=0.8, label=f'P_{i+1} L3 (numerical)')
        
        ax_main.set_xlabel('Time t', fontsize=11)
        ax_main.set_ylabel('Probability', fontsize=11)
        ax_main.set_title('Comparison: L3 Numerical vs L2 Analytical Solution', fontsize=13)
        ax_main.legend(loc='right', fontsize=8, ncol=2)
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xlim(left=0)
        ax_main.set_ylim(0, 1)
        
        # Absolute error plot
        ax_abs = plt.subplot(3, 2, 3)
        for i in range(n_states):
            ax_abs.semilogy(t, abs_error[i, :] + 1e-16, color=colors[i], 
                           linewidth=1.5, label=f'P_{i+1}')
        ax_abs.set_xlabel('Time t', fontsize=11)
        ax_abs.set_ylabel('Absolute Error', fontsize=11)
        ax_abs.set_title('Absolute Error (log scale)', fontsize=12)
        ax_abs.legend(loc='upper right', fontsize=8)
        ax_abs.grid(True, alpha=0.3)
        ax_abs.set_xlim(left=0)
        
        # Relative error plot
        ax_rel = plt.subplot(3, 2, 4)
        for i in range(n_states):
            # Only plot where relative error is meaningful
            mask = rel_error[i, :] > 1e-15
            if np.any(mask):
                ax_rel.semilogy(t[mask], rel_error[i, mask], color=colors[i], 
                               linewidth=1.5, label=f'P_{i+1}')
        ax_rel.set_xlabel('Time t', fontsize=11)
        ax_rel.set_ylabel('Relative Error', fontsize=11)
        ax_rel.set_title('Relative Error (log scale)', fontsize=12)
        ax_rel.legend(loc='upper right', fontsize=8)
        ax_rel.grid(True, alpha=0.3)
        ax_rel.set_xlim(left=0)
        
        # Error distribution histogram
        ax_hist = plt.subplot(3, 2, 5)
        all_errors = abs_error.flatten()
        ax_hist.hist(all_errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax_hist.axvline(comparison_results['global_metrics']['max_abs_error'], 
                       color='red', linestyle='--', linewidth=2, label='Max error')
        ax_hist.axvline(comparison_results['global_metrics']['mean_abs_error'], 
                       color='green', linestyle='--', linewidth=2, label='Mean error')
        ax_hist.set_xlabel('Absolute Error', fontsize=11)
        ax_hist.set_ylabel('Frequency', fontsize=11)
        ax_hist.set_title('Error Distribution', fontsize=12)
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Metrics table
        ax_table = plt.subplot(3, 2, 6)
        ax_table.axis('off')
        
        # Create table data
        metrics = comparison_results['global_metrics']
        max_dev = comparison_results['max_deviation']
        
        table_data = [
            ['Metric', 'Value'],
            ['Step size (h)', f"{comparison_results['step_size']:.6f}"],
            ['Max absolute error', f"{metrics['max_abs_error']:.6e}"],
            ['Mean absolute error', f"{metrics['mean_abs_error']:.6e}"],
            ['RMSE', f"{metrics['rmse']:.6e}"],
            ['Max relative error', f"{metrics['max_rel_error']:.6e}"],
            ['Mean relative error', f"{metrics['mean_rel_error']:.6e}"],
            ['', ''],
            ['Max deviation', ''],
            ['  State', f"S_{max_dev['state']}"],
            ['  Time', f"t = {max_dev['time']:.4f}"],
            ['  Absolute', f"{max_dev['value']:.6e}"],
            ['  Relative', f"{max_dev['relative_value']:.6e}"],
            ['  Interval', f"[{max_dev['interval'][0]:.2f}, {max_dev['interval'][1]:.2f}]"],
        ]
        
        table = ax_table.table(cellText=table_data, loc='center', 
                              cellLoc='left', colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style section headers
        for row_idx in [8]:
            for i in range(2):
                table[(row_idx, i)].set_facecolor('#D9E1F2')
                table[(row_idx, i)].set_text_props(weight='bold')
        
        ax_table.set_title('Error Metrics Summary', fontsize=12, pad=20)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot: {output_file}")
        
        return fig
    
    def generate_comparison_table(self, comparison_results):
        """
        Generate formatted comparison table.
        
        Args:
            comparison_results (dict): Results from compare()
            
        Returns:
            str: Formatted table
        """
        lines = []
        lines.append("=" * 80)
        lines.append("COMPARISON: L3 NUMERICAL vs L2 ANALYTICAL SOLUTION")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Numerical method: Modified Euler (2nd order)")
        lines.append(f"Step size: h = {comparison_results['step_size']:.6f}")
        lines.append("")
        
        # Global metrics
        metrics = comparison_results['global_metrics']
        lines.append("GLOBAL ERROR METRICS:")
        lines.append("-" * 40)
        lines.append(f"  Max absolute error:     {metrics['max_abs_error']:.6e}")
        lines.append(f"  Mean absolute error:    {metrics['mean_abs_error']:.6e}")
        lines.append(f"  RMSE:                   {metrics['rmse']:.6e}")
        lines.append(f"  Max relative error:     {metrics['max_rel_error']:.6e}")
        lines.append(f"  Mean relative error:    {metrics['mean_rel_error']:.6e}")
        lines.append("")
        
        # Per-state metrics
        lines.append("PER-STATE ERROR METRICS:")
        lines.append("-" * 80)
        header = f"{'State':>8} | {'Max Abs':>14} | {'Mean Abs':>14} | {'RMSE':>14} | {'Max Rel':>12}"
        lines.append(header)
        lines.append("-" * 80)
        
        for sm in comparison_results['state_metrics']:
            row = (f"  P_{sm['state']:<4} | {sm['max_abs_error']:>14.6e} | "
                   f"{sm['mean_abs_error']:>14.6e} | {sm['rmse']:>14.6e} | "
                   f"{sm['max_rel_error']:>12.6e}")
            lines.append(row)
        
        lines.append("-" * 80)
        lines.append("")
        
        # Max deviation
        max_dev = comparison_results['max_deviation']
        lines.append("MAXIMUM DEVIATION:")
        lines.append("-" * 40)
        lines.append(f"  State:      S_{max_dev['state']}")
        lines.append(f"  Time:       t = {max_dev['time']:.4f}")
        lines.append(f"  Absolute:   {max_dev['value']:.6e}")
        lines.append(f"  Relative:   {max_dev['relative_value']:.6e}")
        lines.append(f"  Interval:   [{max_dev['interval'][0]:.4f}, {max_dev['interval'][1]:.4f}]")
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)


def compare_solutions(l3_solver, l2_solution=None, output_dir=None):
    """
    Convenience function for full comparison workflow.
    
    Args:
        l3_solver: ModifiedEulerSolver instance
        l2_solution: L2 solution (optional, loads from file if None)
        output_dir: Output directory for plots
        
    Returns:
        dict: Comparison results
    """
    comparator = L3L2Comparator(l3_solver, l2_solution)
    results = comparator.compare()
    
    # Print table
    table = comparator.generate_comparison_table(results)
    print(table)
    
    # Plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_file = os.path.join(output_dir, 'L3_comparison.png')
        comparator.plot_comparison(results, plot_file)
    
    return results


if __name__ == "__main__":
    print("L3-L2 Comparator Test")
    print("=" * 70)
    print("This module requires L2 solution to be available.")
    print("Run L2 first, then use this module for comparison.")

"""
Step size convergence analysis for Modified Euler method.

Analyzes how solution accuracy depends on step size h.
Expected convergence rate: O(h^2) for Modified Euler method.

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-19
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from .modified_euler import ModifiedEulerSolver


class StepConvergenceAnalyzer:
    """
    Analyzer for convergence of Modified Euler method with respect to step size.
    """
    
    def __init__(self, Q, initial_state, t_span=(0, 30)):
        """
        Initialize analyzer.
        
        Args:
            Q (np.ndarray): Intensity matrix
            initial_state (np.ndarray): Initial probability vector
            t_span (tuple): Time interval for integration
        """
        self.Q = Q
        self.initial_state = initial_state
        self.t_span = t_span
        self.results = {}
        
    def run_with_steps(self, h_values, store_steps=500):
        """
        Run solver with multiple step sizes.
        
        Args:
            h_values (list): List of step sizes to test
            store_steps (int): Number of points to store for each run
            
        Returns:
            dict: Results for each step size
        """
        print("Running convergence analysis...")
        print(f"Testing {len(h_values)} step sizes: {h_values}")
        print()
        
        for h in h_values:
            print(f"  h = {h:.6f}...", end=" ")
            
            solver = ModifiedEulerSolver(
                Q=self.Q,
                initial_state=self.initial_state,
                h=h,
                t_span=self.t_span
            )
            
            start_time = time.perf_counter()
            solution = solver.solve(store_steps=store_steps)
            end_time = time.perf_counter()
            
            # Override timing with more accurate measurement
            solution['execution_time'] = end_time - start_time
            
            self.results[h] = {
                'solver': solver,
                'solution': solution
            }
            
            print(f"Done ({solution['n_steps']} steps, {solution['execution_time']*1000:.2f} ms)")
        
        print()
        return self.results
    
    def compute_errors(self, reference_solution, t_test=None):
        """
        Compute errors for all step sizes against reference solution.
        
        Args:
            reference_solution: Reference solution (L2 analytical or very fine grid)
            t_test (np.ndarray): Test time points (uses fine grid if None)
            
        Returns:
            dict: Error metrics for each step size
        """
        if t_test is None:
            # Use common time grid
            t_test = np.linspace(self.t_span[0], self.t_span[1], 200)
        
        errors = {}
        
        # Get reference values
        if hasattr(reference_solution, 'evaluate'):
            y_ref = reference_solution.evaluate(t_test)
        elif isinstance(reference_solution, dict):
            if 't' in reference_solution and 'y' in reference_solution:
                # Interpolate from saved solution
                y_ref = np.zeros((self.Q.shape[0], len(t_test)))
                for i in range(self.Q.shape[0]):
                    y_ref[i, :] = np.interp(
                        t_test,
                        reference_solution['t'],
                        reference_solution['y'][i, :]
                    )
            else:
                raise ValueError("Reference solution dict must contain 't' and 'y' keys")
        else:
            raise ValueError("Invalid reference solution format")
        
        # Compute errors for each step size
        for h, data in self.results.items():
            solver = data['solver']
            y_num = solver.evaluate(t_test)
            
            # Absolute error
            abs_error = np.abs(y_num - y_ref)
            
            # Relative error (avoid division by zero)
            rel_error = np.zeros_like(abs_error)
            mask = y_ref > 1e-10
            rel_error[mask] = abs_error[mask] / y_ref[mask]
            
            # Metrics
            errors[h] = {
                'max_absolute': np.max(abs_error),
                'mean_absolute': np.mean(abs_error),
                'max_relative': np.max(rel_error[mask]) if np.any(mask) else 0,
                'mean_relative': np.mean(rel_error[mask]) if np.any(mask) else 0,
                'rmse': np.sqrt(np.mean(abs_error**2)),
                'execution_time': data['solution']['execution_time'],
                'n_steps': data['solution']['n_steps']
            }
        
        return errors
    
    def estimate_convergence_order(self, errors):
        """
        Estimate empirical convergence order from error data.
        
        For Modified Euler method, expected order is 2 (O(h^2)).
        
        Args:
            errors (dict): Error metrics for each step size
            
        Returns:
            dict: Estimated convergence orders
        """
        h_values = sorted(errors.keys())
        
        if len(h_values) < 2:
            return None
        
        orders = {
            'max_absolute': [],
            'mean_absolute': [],
            'rmse': []
        }
        
        # Compute order from consecutive pairs
        print("\nConvergence order calculation:")
        print(f"{'h1':>10} {'h2':>10} {'h2/h1':>8} | {'err1':>12} {'err2':>12} {'err2/err1':>10} | {'order':>6}")
        print("-" * 75)
        
        for i in range(len(h_values) - 1):
            h1, h2 = h_values[i], h_values[i + 1]
            e1 = errors[h1]
            e2 = errors[h2]
            
            # Order p: error ~ h^p  =>  log(error2/error1) / log(h2/h1) = p
            ratio_h = h2 / h1
            
            for metric in ['max_absolute', 'mean_absolute', 'rmse']:
                if e1[metric] > 1e-15 and e2[metric] > 1e-15:
                    ratio_e = e2[metric] / e1[metric]
                    p = np.log(ratio_e) / np.log(ratio_h)
                    orders[metric].append(p)
                    
                    # Debug output for max_absolute
                    if metric == 'max_absolute':
                        print(f"{h1:>10.4f} {h2:>10.4f} {ratio_h:>8.2f} | {e1[metric]:>12.4e} {e2[metric]:>12.4e} {ratio_e:>10.3f} | {p:>6.3f}")
        
        # Average orders
        avg_orders = {}
        for metric, values in orders.items():
            if values:
                avg_orders[metric] = np.mean(values)
            else:
                avg_orders[metric] = None
        
        return {
            'empirical_orders': orders,
            'average_orders': avg_orders,
            'expected_order': 2.0
        }
    
    def plot_convergence(self, errors, output_file=None):
        """
        Plot convergence analysis results.
        
        Args:
            errors (dict): Error metrics
            output_file (str): Path to save figure
            
        Returns:
            matplotlib.figure.Figure: Created figure
        """
        h_values = sorted(errors.keys())
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.08, hspace=0.3)
        
        max_abs_errors = [errors[h]['max_absolute'] for h in h_values]
        times = [errors[h]['execution_time'] for h in h_values]
        
        # Plot 1: Execution time vs h (log-log)
        ax = axes[0]
        ax.loglog(h_values, times, 'mo-', linewidth=2, markersize=8)
        
        # Annotate points with h values
        for i, h in enumerate(h_values):
            h_str = f'{h:g}'
            # Alternate position: even above, odd below
            if i % 2 == 0:
                xytext = (0, 12)
                va = 'bottom'
            else:
                xytext = (0, -12)
                va = 'top'
            ax.annotate(f'h={h_str}', (h_values[i], times[i]),
                       fontsize=10, ha='center', va=va,
                       xytext=xytext, textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='magenta', alpha=0.8))
        
        ax.set_xlabel('Step size h', fontsize=11)
        ax.set_ylabel('Execution time (s)', fontsize=11)
        ax.set_title('Execution Time vs Step Size', fontsize=12)
        ax.grid(True, alpha=0.3, which='both')
        ax.margins(x=0.15, y=0.15)
        
        # Ensure more ticks are shown on log axes
        ax.minorticks_on()
        ax.tick_params(which='both', labelsize=9)
        
        # Plot 2: Accuracy vs Computational cost
        ax = axes[1]
        ax.loglog(times, max_abs_errors, 'co-', linewidth=2, markersize=8)
        
        # Annotate points with h values and error values
        for i, h in enumerate(h_values):
            h_str = f'{h:g}'
            
            # Label: h value + error value (scientific notation)
            err_str = f'{max_abs_errors[i]:.2e}'
            label = f'h={h_str}\nerr={err_str}'
            
            # Alternate text position to avoid overlap
            if i % 2 == 0:
                xytext = (12, 0)
                ha = 'left'
            else:
                xytext = (-12, 0)
                ha = 'right'
            
            ax.annotate(label, (times[i], max_abs_errors[i]),
                       fontsize=9, ha=ha, va='center',
                       xytext=xytext, textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='cyan', alpha=0.8))
        
        ax.set_xlabel('Execution time (s)', fontsize=11)
        ax.set_ylabel('Max absolute error', fontsize=11)
        ax.set_title('Accuracy vs Computational Cost', fontsize=12)
        ax.grid(True, alpha=0.3, which='both')
        ax.margins(x=0.25, y=0.2)
        
        # Ensure more ticks are shown on log axes
        ax.minorticks_on()
        ax.tick_params(which='both', labelsize=9)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved convergence plot: {output_file}")
        
        return fig
    
    def generate_convergence_table(self, errors, convergence_order=None):
        """
        Generate formatted convergence table.
        
        Args:
            errors (dict): Error metrics
            convergence_order (dict): Convergence order estimates
            
        Returns:
            str: Formatted table
        """
        h_values = sorted(errors.keys())
        
        lines = []
        lines.append("=" * 90)
        lines.append("CONVERGENCE ANALYSIS: MODIFIED EULER METHOD")
        lines.append("=" * 90)
        lines.append("")
        
        # Header
        header = f"{'Step h':>12} | {'Steps':>8} | {'Time (ms)':>10} | {'Max Abs Err':>14} | {'RMSE':>14} | {'Max Rel Err':>12}"
        lines.append(header)
        lines.append("-" * 90)
        
        # Data rows
        for h in h_values:
            e = errors[h]
            row = (f"{h:>12.6f} | {e['n_steps']:>8} | {e['execution_time']*1000:>10.2f} | "
                   f"{e['max_absolute']:>14.6e} | {e['rmse']:>14.6e} | {e['max_relative']:>12.6e}")
            lines.append(row)
        
        lines.append("-" * 90)
        lines.append("")
        
        # Convergence order
        if convergence_order and convergence_order['average_orders']:
            lines.append("CONVERGENCE ORDER ESTIMATES:")
            lines.append(f"  Expected order (theoretical): O(h²) = 2.0")
            for metric, order in convergence_order['average_orders'].items():
                if order is not None:
                    lines.append(f"  Empirical order ({metric}): {order:.3f}")
            lines.append("")
        
        lines.append("=" * 90)
        
        return "\n".join(lines)


def analyze_convergence(Q, initial_state, reference_solution, 
                       h_values=None, t_span=(0, 30), output_dir=None):
    """
    Convenience function for full convergence analysis.
    
    Args:
        Q: Intensity matrix
        initial_state: Initial probabilities
        reference_solution: Reference for error computation
        h_values: List of step sizes (default: [0.1, 0.05, 0.025, 0.01])
        t_span: Time interval
        output_dir: Directory for output files
        
    Returns:
        dict: Analysis results
    """
    if h_values is None:
        h_values = [0.1, 0.05, 0.025, 0.01]
    
    analyzer = StepConvergenceAnalyzer(Q, initial_state, t_span)
    
    # Run with different steps
    analyzer.run_with_steps(h_values)
    
    # Compute errors
    errors = analyzer.compute_errors(reference_solution)
    
    # Estimate convergence order
    order = analyzer.estimate_convergence_order(errors)
    
    # Generate table
    table = analyzer.generate_convergence_table(errors, order)
    print(table)
    
    # Plot
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        plot_file = os.path.join(output_dir, 'L3_convergence.png')
        analyzer.plot_convergence(errors, plot_file)
    
    return {
        'analyzer': analyzer,
        'errors': errors,
        'convergence_order': order,
        'table': table
    }


if __name__ == "__main__":
    # Test with simple system
    Q = np.array([[-0.5, 0.3, 0.2],
                  [0.4, -0.7, 0.3],
                  [0.0, 0.0, 0.0]])
    P0 = np.array([1.0, 0.0, 0.0])
    
    print("Step Convergence Analysis Test")
    print("=" * 70)
    
    # Create a reference solution with very small step
    ref_solver = ModifiedEulerSolver(Q, P0, h=0.001, t_span=(0, 10))
    ref_result = ref_solver.solve(store_steps=500)
    reference = {'t': ref_result['t'], 'y': ref_result['y']}
    
    # Analyze convergence
    results = analyze_convergence(
        Q, P0, reference,
        h_values=[0.1, 0.05, 0.025, 0.01],
        t_span=(0, 10)
    )

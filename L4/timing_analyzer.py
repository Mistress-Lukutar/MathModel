"""
Timing Analyzer for Lab 4.

Analyzes computational time costs for different step sizes
and computes efficiency metrics.

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-19
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Callable


class TimingAnalyzer:
    """
    Analyzer for computational timing and efficiency.
    
    Measures execution time for different step sizes and
    computes efficiency metrics (accuracy per unit time).
    """
    
    def __init__(self):
        """Initialize timing analyzer."""
        self.timing_results = []
    
    def measure_execution_time(self, solver_func: Callable, 
                               *args, **kwargs) -> Dict:
        """
        Measure execution time of a solver function.
        
        Args:
            solver_func: Function to measure
            *args, **kwargs: Arguments for the function
            
        Returns:
            Dict with timing results
        """
        # Warm-up run (to exclude JIT compilation effects)
        _ = solver_func(*args, **kwargs)
        
        # Timed runs
        n_runs = 3
        times = []
        
        for _ in range(n_runs):
            start = time.perf_counter()
            result = solver_func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            'result': result,
            'times': times,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times)
        }
    
    def analyze_timing_vs_step(self, convergence_results: List[Dict]) -> Dict:
        """
        Analyze relationship between step size and execution time.
        
        Args:
            convergence_results: Results from ConvergenceAnalyzer
            
        Returns:
            Dict with timing analysis
        """
        steps = []
        times = []
        n_steps_list = []
        
        for r in convergence_results:
            steps.append(r['h'])
            times.append(r['execution_time'])
            n_steps_list.append(r['n_steps'])
        
        steps = np.array(steps)
        times = np.array(times)
        
        # Fit time ~ C / h (should be linear relationship)
        # Use linear fit in log-log scale
        log_h = np.log(steps)
        log_t = np.log(times)
        
        # Linear regression
        A = np.vstack([log_h, np.ones(len(log_h))]).T
        slope, intercept = np.linalg.lstsq(A, log_t, rcond=None)[0]
        
        # Theoretical: time ~ 1/h, so slope should be -1
        return {
            'steps': steps,
            'times': times,
            'n_steps': n_steps_list,
            'complexity_slope': slope,
            'complexity_intercept': intercept,
            'theoretical_slope': -1.0,
            'time_per_step': times / np.array(n_steps_list)
        }
    
    def compute_efficiency_metrics(self, convergence_results: List[Dict]) -> List[Dict]:
        """
        Compute efficiency metrics (accuracy per unit time).
        
        Args:
            convergence_results: Results from ConvergenceAnalyzer
            
        Returns:
            List of efficiency metrics for each step size
        """
        metrics = []
        
        for r in convergence_results:
            if r['max_error'] is not None and r['execution_time'] > 0:
                # Efficiency: 1 / (error * time) - higher is better
                efficiency = 1.0 / (r['max_error'] * r['execution_time'])
                
                # Accuracy per time: -log(error) / time
                accuracy_per_time = -np.log10(r['max_error']) / r['execution_time']
                
                metric = {
                    'h': r['h'],
                    'error': r['max_error'],
                    'time': r['execution_time'],
                    'efficiency': efficiency,
                    'accuracy_per_time': accuracy_per_time,
                    'n_steps': r['n_steps']
                }
                metrics.append(metric)
        
        return metrics
    
    def find_optimal_step(self, convergence_results: List[Dict],
                         criterion: str = 'balanced') -> Dict:
        """
        Find optimal step size based on given criterion.
        
        Args:
            convergence_results: Results from ConvergenceAnalyzer
            criterion: 'accuracy', 'speed', or 'balanced'
            
        Returns:
            Dict with optimal step information
        """
        if not convergence_results:
            return None
        
        if criterion == 'accuracy':
            # Minimize error
            best = min(convergence_results, key=lambda x: x['max_error'] if x['max_error'] else float('inf'))
            return {'h': best['h'], 'error': best['max_error'], 
                    'time': best['execution_time'], 'criterion': 'accuracy'}
        
        elif criterion == 'speed':
            # Minimize time
            best = min(convergence_results, key=lambda x: x['execution_time'])
            return {'h': best['h'], 'error': best['max_error'], 
                    'time': best['execution_time'], 'criterion': 'speed'}
        
        else:  # balanced
            # Best efficiency (accuracy per unit time)
            metrics = self.compute_efficiency_metrics(convergence_results)
            if metrics:
                best = max(metrics, key=lambda x: x['efficiency'])
                return {'h': best['h'], 'error': best['error'], 
                        'time': best['time'], 'criterion': 'balanced',
                        'efficiency': best['efficiency']}
            return None
    
    def print_timing_report(self, convergence_results: List[Dict]):
        """Print timing analysis report."""
        if not convergence_results:
            print("No results available.")
            return
        
        timing_analysis = self.analyze_timing_vs_step(convergence_results)
        efficiency_metrics = self.compute_efficiency_metrics(convergence_results)
        
        print("\n" + "=" * 70)
        print("TIMING ANALYSIS REPORT")
        print("=" * 70)
        
        print("\nTiming vs Step Size:")
        print(f"{'h':<12}{'Time (ms)':<15}{'N steps':<12}{'Time/step (μs)':<18}")
        print("-" * 60)
        
        for i, h in enumerate(timing_analysis['steps']):
            time_ms = timing_analysis['times'][i] * 1000
            n_steps = timing_analysis['n_steps'][i]
            time_per_step = timing_analysis['time_per_step'][i] * 1e6  # microseconds
            print(f"{h:<12.4f}{time_ms:<15.2f}{n_steps:<12}{time_per_step:<18.2f}")
        
        print(f"\nComplexity Analysis:")
        print(f"  Fitted relationship: time ~ h^{timing_analysis['complexity_slope']:.2f}")
        print(f"  Theoretical: time ~ h^(-1.00)")
        
        if abs(timing_analysis['complexity_slope'] - (-1.0)) < 0.2:
            print("  ✓ Matches theoretical O(1/h) complexity!")
        
        if efficiency_metrics:
            print(f"\nEfficiency Metrics:")
            print(f"{'h':<12}{'Error':<18}{'Time (ms)':<15}{'Efficiency':<15}")
            print("-" * 60)
            for m in efficiency_metrics:
                print(f"{m['h']:<12.4f}{m['error']:<18.6e}{m['time']*1000:<15.2f}{m['efficiency']:<15.2e}")
            
            # Optimal steps
            for criterion in ['accuracy', 'speed', 'balanced']:
                optimal = self.find_optimal_step(convergence_results, criterion)
                if optimal:
                    print(f"\n  Optimal for {criterion}: h = {optimal['h']}")
        
        print("\n" + "=" * 70)
    
    def plot_timing_analysis(self, convergence_results: List[Dict],
                            output_dir: Optional[str] = None,
                            suffix: str = "") -> str:
        """
        Plot timing analysis results.
        
        Args:
            convergence_results: Convergence analysis results
            output_dir: Directory to save figure
            suffix: Suffix for filename (e.g., 'coarse' or 'fine')
            
        Returns:
            Path to saved figure
        """
        import matplotlib.pyplot as plt
        import os
        
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(script_dir)
            output_dir = os.path.join(root_dir, 'Output')
        
        os.makedirs(output_dir, exist_ok=True)
        
        timing_analysis = self.analyze_timing_vs_step(convergence_results)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        steps = timing_analysis['steps']
        times = timing_analysis['times']
        
        # Left: Time vs step
        ax = axes[0]
        ax.plot(steps, times * 1000, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Step Size (h)', fontsize=11)
        ax.set_ylabel('Execution Time (ms)', fontsize=11)
        ax.set_title('Execution Time vs Step Size', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Right: Time vs 1/h (should be linear)
        ax = axes[1]
        inv_h = 1.0 / steps
        ax.plot(inv_h, times * 1000, 'ro-', linewidth=2, markersize=8)
        ax.set_xlabel('1/h', fontsize=11)
        ax.set_ylabel('Execution Time (ms)', fontsize=11)
        ax.set_title('Execution Time vs 1/h (Linear Check)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Fit line
        if len(steps) >= 2:
            A = np.vstack([inv_h, np.ones(len(inv_h))]).T
            slope, intercept = np.linalg.lstsq(A, times * 1000, rcond=None)[0]
            inv_h_fit = np.linspace(min(inv_h), max(inv_h), 100)
            time_fit = slope * inv_h_fit + intercept
            ax.plot(inv_h_fit, time_fit, 'g--', linewidth=1.5, 
                   alpha=0.7, label=f'Fit: t = {slope:.3f}/h + {intercept:.3f}')
            ax.legend(fontsize=9)
        
        title_suffix = f" ({suffix})" if suffix else ""
        plt.suptitle(f'Timing Analysis - Modified Euler Method{title_suffix}', fontsize=13)
        plt.tight_layout()
        
        filename = f'L4_timing_{suffix}.png' if suffix else 'L4_timing.png'
        output_file = os.path.join(output_dir, filename)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved timing plot: {output_file}")
        return output_file


if __name__ == "__main__":
    # Example usage
    # Mock convergence results
    mock_results = [
        {'h': 0.04, 'n_steps': 750, 'execution_time': 0.05, 'max_error': 1e-3},
        {'h': 0.02, 'n_steps': 1500, 'execution_time': 0.10, 'max_error': 2.5e-4},
        {'h': 0.01, 'n_steps': 3000, 'execution_time': 0.20, 'max_error': 6.25e-5},
        {'h': 0.005, 'n_steps': 6000, 'execution_time': 0.40, 'max_error': 1.56e-5},
    ]
    
    analyzer = TimingAnalyzer()
    analyzer.print_timing_report(mock_results)

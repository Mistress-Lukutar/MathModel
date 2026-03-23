"""
Accuracy Analyzer for Lab 4.

Compares numerical solution (L3) with analytical solution (L2)
and computes various error metrics.

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-19
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class AccuracyAnalyzer:
    """
    Analyzer for comparing numerical and analytical solutions.
    
    Computes absolute error, relative error, RMSE, and identifies
    intervals of maximum deviation.
    """
    
    def __init__(self, t_num: np.ndarray, y_num: np.ndarray,
                 t_anal: np.ndarray, y_anal: np.ndarray,
                 state_names: Optional[List[str]] = None):
        """
        Initialize analyzer with numerical and analytical solutions.
        
        Args:
            t_num: Time points for numerical solution
            y_num: Numerical solution (n_states × n_points)
            t_anal: Time points for analytical solution
            y_anal: Analytical solution (n_states × n_points)
            state_names: Optional list of state names
        """
        self.t_num = np.array(t_num)
        self.y_num = np.array(y_num)
        self.t_anal = np.array(t_anal)
        self.y_anal = np.array(y_anal)
        self.n_states = y_num.shape[0]
        
        if state_names is None:
            self.state_names = [f"P_{i+1}" for i in range(self.n_states)]
        else:
            self.state_names = state_names
        
        # Interpolate analytical solution to numerical time points
        self._interpolate_analytical()
        
        # Compute errors
        self._compute_errors()
    
    def _interpolate_analytical(self):
        """Interpolate analytical solution to numerical time grid."""
        self.y_anal_interp = np.zeros_like(self.y_num)
        
        for i in range(self.n_states):
            self.y_anal_interp[i, :] = np.interp(
                self.t_num, self.t_anal, self.y_anal[i, :]
            )
    
    def _compute_errors(self):
        """Compute absolute and relative errors."""
        # Absolute error
        self.abs_error = np.abs(self.y_num - self.y_anal_interp)
        
        # Relative error (avoid division by zero)
        self.rel_error = np.zeros_like(self.abs_error)
        self.mask = self.y_anal_interp > 1e-10
        self.rel_error[self.mask] = self.abs_error[self.mask] / self.y_anal_interp[self.mask]
    
    def compute_global_metrics(self) -> Dict:
        """
        Compute global error metrics across all states and time.
        
        Returns:
            Dict with global error metrics
        """
        return {
            'max_abs_error': np.max(self.abs_error),
            'mean_abs_error': np.mean(self.abs_error),
            'max_rel_error': np.max(self.rel_error[self.mask]) if np.any(self.mask) else 0,
            'mean_rel_error': np.mean(self.rel_error[self.mask]) if np.any(self.mask) else 0,
            'rmse': np.sqrt(np.mean(self.abs_error**2))
        }
    
    def compute_state_metrics(self) -> List[Dict]:
        """
        Compute error metrics for each state.
        
        Returns:
            List of dicts with per-state metrics
        """
        metrics = []
        
        for i in range(self.n_states):
            mask_state = self.y_anal_interp[i, :] > 1e-10
            
            state_metrics = {
                'state': i + 1,
                'name': self.state_names[i],
                'max_abs_error': np.max(self.abs_error[i, :]),
                'mean_abs_error': np.mean(self.abs_error[i, :]),
                'max_rel_error': np.max(self.rel_error[i, mask_state]) if np.any(mask_state) else 0,
                'mean_rel_error': np.mean(self.rel_error[i, mask_state]) if np.any(mask_state) else 0,
                'rmse': np.sqrt(np.mean(self.abs_error[i, :]**2))
            }
            metrics.append(state_metrics)
        
        return metrics
    
    def find_max_deviation_interval(self, window_size: int = 10) -> Dict:
        """
        Find time interval with maximum deviation.
        
        Args:
            window_size: Number of points to consider as interval
            
        Returns:
            Dict with interval information
        """
        # Compute sum of errors across all states at each time point
        total_error = np.sum(self.abs_error, axis=0)
        
        # Find window with maximum average error
        max_avg_error = 0
        max_interval_idx = 0
        
        for i in range(len(self.t_num) - window_size + 1):
            window_avg = np.mean(total_error[i:i+window_size])
            if window_avg > max_avg_error:
                max_avg_error = window_avg
                max_interval_idx = i
        
        start_idx = max_interval_idx
        end_idx = min(max_interval_idx + window_size, len(self.t_num))
        
        # Find states with max error in this interval
        interval_errors = self.abs_error[:, start_idx:end_idx]
        max_error_states = np.argsort(np.max(interval_errors, axis=1))[-3:][::-1]
        
        return {
            't_start': self.t_num[start_idx],
            't_end': self.t_num[end_idx - 1],
            'max_avg_error': max_avg_error,
            'max_error_states': [int(s + 1) for s in max_error_states],
            'start_idx': start_idx,
            'end_idx': end_idx
        }
    
    def get_time_of_max_error(self) -> Dict:
        """
        Get time point where maximum error occurs for each state.
        
        Returns:
            Dict with time points of maximum errors
        """
        result = {}
        
        for i in range(self.n_states):
            max_idx = np.argmax(self.abs_error[i, :])
            result[f"P_{i+1}"] = {
                't': self.t_num[max_idx],
                'max_error': self.abs_error[i, max_idx],
                'index': int(max_idx)
            }
        
        return result
    
    def get_all_metrics(self) -> Dict:
        """
        Get complete set of accuracy metrics.
        
        Returns:
            Dict with all metrics
        """
        return {
            'global': self.compute_global_metrics(),
            'per_state': self.compute_state_metrics(),
            'max_deviation_interval': self.find_max_deviation_interval(),
            'time_of_max_error': self.get_time_of_max_error()
        }
    
    def print_summary(self):
        """Print summary of accuracy analysis."""
        metrics = self.get_all_metrics()
        
        print("\n" + "=" * 70)
        print("ACCURACY ANALYSIS SUMMARY")
        print("=" * 70)
        
        print("\nGlobal Metrics:")
        print(f"  Max absolute error:  {metrics['global']['max_abs_error']:.6e}")
        print(f"  Mean absolute error: {metrics['global']['mean_abs_error']:.6e}")
        print(f"  Max relative error:  {metrics['global']['max_rel_error']:.6e}")
        print(f"  RMSE:                {metrics['global']['rmse']:.6e}")
        
        print("\nPer-State Metrics:")
        print(f"{'State':<10}{'Max Abs':<15}{'Max Rel':<15}{'RMSE':<15}")
        print("-" * 55)
        for m in metrics['per_state']:
            print(f"{m['name']:<10}{m['max_abs_error']:<15.6e}{m['max_rel_error']:<15.6e}{m['rmse']:<15.6e}")
        
        interval = metrics['max_deviation_interval']
        print(f"\nInterval of Maximum Deviation:")
        print(f"  Time range: [{interval['t_start']:.4f}, {interval['t_end']:.4f}]")
        print(f"  States with max error: {interval['max_error_states']}")
        
        print("\n" + "=" * 70)


if __name__ == "__main__":
    # Example usage
    t = np.linspace(0, 10, 100)
    y_num = np.array([np.sin(t), np.cos(t)])
    y_anal = np.array([np.sin(t) * 0.99, np.cos(t) * 1.01])  # slightly different
    
    analyzer = AccuracyAnalyzer(t, y_num, t, y_anal)
    analyzer.print_summary()

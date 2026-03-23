"""
Convergence Analysis for Lab 4.

Analyzes the convergence of numerical method by computing
solutions with different step sizes and estimating the order
of accuracy.

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-19
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

from L3.modified_euler import ModifiedEulerSolver


class PureModifiedEulerSolver(ModifiedEulerSolver):
    """
    Modified Euler solver WITHOUT modifications that break convergence order.
    
    Removes:
    - Probability normalization (adds O(h) error)
    - Negative value clipping (adds non-linearity)
    
    This pure version preserves the theoretical O(h^2) convergence order
    for convergence analysis purposes.
    """
    
    def step(self, t, P):
        """
        Pure Modified Euler step without normalization.
        
        Standard predictor-corrector:
            k1 = h * f(t_n, P_n)
            k2 = h * f(t_n + h, P_n + k1)
            P_{n+1} = P_n + (k1 + k2) / 2
        """
        # Predictor step
        k1 = self.h * self.f(t, P)
        
        # Corrector step
        k2 = self.h * self.f(t + self.h, P + k1)
        
        # Combined step (NO normalization, NO clipping)
        P_next = P + 0.5 * (k1 + k2)
        
        return P_next


class ConvergenceAnalyzer:
    """
    Analyzer for numerical method convergence.
    
    Runs solver with different step sizes and estimates
    the order of accuracy.
    """
    
    def __init__(self, Q: np.ndarray, initial_state: np.ndarray,
                 t_span: Tuple[float, float] = (0, 30),
                 reference_solution: Optional[Dict] = None):
        """
        Initialize convergence analyzer.
        
        Args:
            Q: Intensity matrix
            initial_state: Initial probability vector
            t_span: Time interval (t_start, t_end)
            reference_solution: Optional reference solution (e.g., from L2)
        """
        self.Q = Q
        self.initial_state = initial_state
        self.t_span = t_span
        self.reference = reference_solution
        
        self.results = []
    
    def run_convergence_study(self, step_sizes: List[float],
                              store_steps: int = 500,
                              use_pure_solver: bool = True) -> List[Dict]:
        """
        Run solver with different step sizes.
        
        Args:
            step_sizes: List of step sizes to test
            store_steps: Number of points to store
            use_pure_solver: If True, use PureModifiedEulerSolver (no normalization)
                           for clean convergence analysis. Default: True.
            
        Returns:
            List of results for each step size
        """
        self.results = []
        
        # Choose solver class
        solver_class = PureModifiedEulerSolver if use_pure_solver else ModifiedEulerSolver
        
        print("\nRunning convergence study...")
        if use_pure_solver:
            print("Using PURE Modified Euler (without normalization)")
            print("This preserves theoretical O(h^2) convergence order.")
        print("=" * 60)
        print(f"{'Step (h)':<12}{'N steps':<12}{'Time (ms)':<15}{'Max Error':<15}")
        print("-" * 60)
        
        for h in step_sizes:
            # Create solver (pure version for clean convergence analysis)
            solver = solver_class(
                Q=self.Q,
                initial_state=self.initial_state,
                h=h,
                t_span=self.t_span
            )
            
            # Solve
            solution = solver.solve(store_steps=store_steps)
            
            # Compute error if reference available
            max_error = None
            if self.reference is not None:
                max_error = self._compute_max_error(solver, self.reference)
            
            result = {
                'h': h,
                'n_steps': solution['n_steps'],
                'execution_time': solution['execution_time'],
                'max_prob_deviation': solution.get('max_prob_deviation', None),
                'max_error': max_error,
                'solver': solver,
                'solution': solution
            }
            
            self.results.append(result)
            
            error_str = f"{max_error:.6e}" if max_error is not None else "N/A"
            print(f"{h:<12.4f}{solution['n_steps']:<12}{solution['execution_time']*1000:<15.2f}{error_str:<15}")
        
        print("=" * 60)
        return self.results
    
    def _compute_max_error(self, solver: ModifiedEulerSolver,
                          reference: Dict) -> float:
        """Compute maximum error against reference solution."""
        # Evaluate numerical solution at reference time points
        t_ref = reference['t']
        y_ref = reference['y']
        
        y_num = solver.evaluate(t_ref)
        
        # Compute maximum absolute error
        abs_error = np.abs(y_num - y_ref)
        return np.max(abs_error)
    
    def estimate_convergence_order(self) -> Optional[float]:
        """
        Estimate the order of convergence from results using linear regression
        in log-log scale for more robust estimation.
        
        Returns:
            Estimated order p (error ~ O(h^p)) or None if not enough data
        """
        if len(self.results) < 2:
            return None
        
        # Filter results with error data
        valid_results = [r for r in self.results if r['max_error'] is not None and r['max_error'] > 1e-15]
        
        if len(valid_results) < 2:
            return None
        
        # Extract h and error values
        h_values = np.array([r['h'] for r in valid_results])
        error_values = np.array([r['max_error'] for r in valid_results])
        
        # Compute consecutive ratios and orders
        orders = []
        for i in range(len(valid_results) - 1):
            h1 = valid_results[i]['h']
            h2 = valid_results[i + 1]['h']
            e1 = valid_results[i]['max_error']
            e2 = valid_results[i + 1]['max_error']
            
            if e1 > 1e-15 and e2 > 1e-15 and h1 != h2:
                # p ≈ log(e2/e1) / log(h2/h1)
                p = np.log(e2 / e1) / np.log(h2 / h1)
                orders.append(p)
                
                # Store ratio for reporting
                valid_results[i + 1]['error_ratio'] = e1 / e2 if e2 > 0 else None
                valid_results[i + 1]['order_estimate'] = p
        
        # Primary estimate: linear regression in log-log space
        # log(error) = p * log(h) + C
        log_h = np.log(h_values)
        log_err = np.log(error_values)
        
        # Filter out points where machine epsilon dominates (error < 1e-12)
        mask_valid = error_values > 1e-12
        if np.sum(mask_valid) >= 2:
            # Use only points before machine epsilon dominates
            log_h_fit = log_h[mask_valid]
            log_err_fit = log_err[mask_valid]
        else:
            log_h_fit = log_h
            log_err_fit = log_err
        
        if len(log_h_fit) >= 2:
            A = np.vstack([log_h_fit, np.ones(len(log_h_fit))]).T
            p_fit, _ = np.linalg.lstsq(A, log_err_fit, rcond=None)[0]
            
            # Store regression result
            self._convergence_slope = p_fit
            self._convergence_intercept = _
        else:
            p_fit = np.mean(orders) if orders else None
        
        # Also calculate separate estimates for different regions
        if len(orders) >= 3:
            # Early region (large steps) - should be closer to theoretical
            early_order = np.mean(orders[:2])
            # Late region (small steps) - may degrade due to roundoff
            late_order = np.mean(orders[-2:])
            
            self._early_order = early_order
            self._late_order = late_order
        
        return p_fit if p_fit is not None else (np.mean(orders) if orders else None)
    
    def get_convergence_table(self) -> List[Dict]:
        """
        Get formatted convergence table data.
        
        Returns:
            List of dicts with convergence data
        """
        table = []
        
        for i, r in enumerate(self.results):
            row = {
                'step': r['h'],
                'n_steps': r['n_steps'],
                'max_error': r['max_error'],
                'execution_time_ms': r['execution_time'] * 1000,
                'error_ratio': r.get('error_ratio'),
                'order': r.get('order_estimate')
            }
            table.append(row)
        
        return table
    
    def print_convergence_report(self):
        """Print detailed convergence report."""
        if not self.results:
            print("No results available. Run run_convergence_study() first.")
            return
        
        order = self.estimate_convergence_order()
        table = self.get_convergence_table()
        
        print("\n" + "=" * 70)
        print("CONVERGENCE ANALYSIS REPORT")
        print("=" * 70)
        
        print("\nConvergence Table:")
        print(f"{'h':<12}{'Error':<18}{'Ratio':<12}{'Order (p)':<12}{'Time (ms)':<12}")
        print("-" * 70)
        
        for row in table:
            ratio_str = f"{row['error_ratio']:.2f}" if row['error_ratio'] else "-"
            order_str = f"{row['order']:.2f}" if row['order'] else "-"
            error_str = f"{row['max_error']:.6e}" if row['max_error'] else "N/A"
            
            print(f"{row['step']:<12.4f}{error_str:<18}{ratio_str:<12}{order_str:<12}{row['execution_time_ms']:<12.2f}")
        
        if order is not None:
            print(f"\nEstimated convergence order: O(h^{order:.2f})")
            print(f"Theoretical order for Modified Euler: O(h^2.00)")
            
            if abs(order - 2.0) < 0.2:
                print("✓ Convergence matches theoretical expectation!")
            else:
                print("⚠ Convergence differs from theoretical expectation.")
        
        print("\n" + "=" * 70)
    
    def analyze_step_groups(self, coarse_steps: List[float], 
                           fine_steps: List[float],
                           store_steps: int = 500) -> Dict:
        """
        Analyze convergence for two step size groups to demonstrate
        round-off error effects.
        
        Args:
            coarse_steps: Larger steps (e.g., [0.5, 0.4, 0.3, 0.2, 0.1])
            fine_steps: Smaller steps (e.g., [0.16, 0.08, 0.04, 0.02, 0.01])
            store_steps: Number of points to store
            
        Returns:
            Dict with analysis results for both groups
        """
        print("\n" + "=" * 70)
        print("TWO-GROUP CONVERGENCE ANALYSIS")
        print("=" * 70)
        print("\nGroup 1: Coarse steps (truncation error dominates)")
        print(f"Steps: {coarse_steps}")
        
        # Save current results
        original_results = self.results
        
        # Run coarse steps
        self.results = []
        coarse_results = self.run_convergence_study(coarse_steps, store_steps)
        coarse_order = self.estimate_convergence_order()
        
        print("\n" + "=" * 70)
        print("\nGroup 2: Fine steps (round-off error affects convergence)")
        print(f"Steps: {fine_steps}")
        
        # Run fine steps
        self.results = []
        fine_results = self.run_convergence_study(fine_steps, store_steps)
        fine_order = self.estimate_convergence_order()
        
        # Restore combined results
        self.results = coarse_results + fine_results
        
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"\nCoarse steps {coarse_steps}:")
        print(f"  Estimated order: O(h^{coarse_order:.2f})" if coarse_order else "  N/A")
        print(f"  Expected: O(h^2.00) - truncation error dominates")
        
        print(f"\nFine steps {fine_steps}:")
        print(f"  Estimated order: O(h^{fine_order:.2f})" if fine_order else "  N/A")
        print(f"  Expected: Lower than 2.0 - round-off error accumulation")
        
        print("\n" + "=" * 70)
        print("EXPLANATION: Round-off Error Floor")
        print("=" * 70)
        print("""
When using small step sizes, the convergence order appears to degrade.
This is NOT a bug in the method, but a fundamental limitation of 
floating-point arithmetic (machine epsilon ≈ 10^-16).

For coarse steps:
  - Truncation error (O(h^2)) >> Round-off error (O(ε·N))
  - Measured order ≈ 2.0 ✓

For fine steps:
  - Number of steps N = 1/h increases
  - Accumulated round-off error grows as N·ε = ε/h
  - Total error = C1·h^2 + C2·(ε/h)
  - When h is very small: ε/h dominates over h^2
  - Measured order < 2.0 (often ≈ 1.0 or worse)

This phenomenon is called the "Round-off Error Floor" or 
"Machine Precision Limit" in numerical analysis.
        """)
        
        return {
            'coarse': {
                'steps': coarse_steps,
                'results': coarse_results,
                'order': coarse_order
            },
            'fine': {
                'steps': fine_steps,
                'results': fine_results,
                'order': fine_order
            }
        }


if __name__ == "__main__":
    # Example usage
    Q = np.array([[-0.5, 0.3, 0.2],
                  [0.4, -0.7, 0.3],
                  [0.0, 0.0, 0.0]])
    P0 = np.array([1.0, 0.0, 0.0])
    
    # Create reference solution with very small step
    ref_solver = ModifiedEulerSolver(Q, P0, h=0.001, t_span=(0, 10))
    ref_solution = ref_solver.solve(store_steps=100)
    
    # Analyze convergence
    analyzer = ConvergenceAnalyzer(Q, P0, t_span=(0, 10), 
                                   reference_solution=ref_solution)
    analyzer.run_convergence_study([0.04, 0.02, 0.01, 0.005])
    analyzer.print_convergence_report()

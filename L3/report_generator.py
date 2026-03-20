"""
Report generator for Lab 3.

Generates comprehensive text report with:
- Method description
- Results tables
- Error analysis

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-19
"""

import os
import numpy as np
from datetime import datetime


class L3ReportGenerator:
    """
    Generator for Lab 3 reports.
    """
    
    def __init__(self, solver, comparison_results=None):
        """
        Initialize report generator.
        
        Args:
            solver: ModifiedEulerSolver instance
            comparison_results: Results from L3L2Comparator
        """
        self.solver = solver
        self.comparison_results = comparison_results
    
    def generate(self):
        """
        Generate full report.
        
        Returns:
            str: Report text
        """
        sections = [
            self._generate_header(),
            self._generate_theory(),
            self._generate_input_data(),
            self._generate_results(),
            self._generate_error_analysis(),
            self._generate_conclusions(),
            self._generate_footer()
        ]
        
        return "\n\n".join(sections)
    
    def _generate_header(self):
        """Generate report header."""
        lines = [
            "=" * 80,
            "LABORATORY WORK 3",
            "Numerical Solution of Linear Differential Equations",
            "=" * 80,
            "",
            f"Variant: 8. Modified Euler Method",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "=" * 80,
        ]
        return "\n".join(lines)
    
    def _generate_theory(self):
        """Generate theoretical section."""
        lines = [
            "1. THEORETICAL BACKGROUND",
            "=" * 80,
            "",
            "1.1 Problem Statement",
            "-" * 40,
            "System of linear differential equations (Kolmogorov forward equations):",
            "",
            "    dP/dt = P · Q",
            "",
            "where:",
            "  - P(t) = [P_1(t), P_2(t), ..., P_n(t)] - probability vector",
            "  - Q - intensity matrix (infinitesimal generator)",
            "  - Initial condition: P(0) = P_0",
            "",
            "1.2 Modified Euler Method (Heun's Method)",
            "-" * 40,
            "The Modified Euler method is a predictor-corrector scheme:",
            "",
            "    Predictor (Euler step):",
            "        k_1 = h · f(t_j, P_j)",
            "",
            "    Corrector:",
            "        k_2 = h · f(t_j + h, P_j + k_1)",
            "",
            "    Update:",
            "        P_{j+1} = P_j + (k_1 + k_2) / 2",
            "        t_{j+1} = t_j + h",
            "",
            "Properties:",
            "  - Order of accuracy: O(h²) - second order",
            "  - Type: Single-step, two-stage method",
            "  - Stability: Conditionally stable",
            "  - Computational cost: 2 function evaluations per step",
            "",
            "For our system f(t, P) = P · Q (autonomous, so t is not used).",
            "",
        ]
        return "\n".join(lines)
    
    def _generate_input_data(self):
        """Generate input data section."""
        lines = [
            "2. INPUT DATA",
            "=" * 80,
            "",
            f"Number of states: {self.solver.n_states}",
            f"Initial state: P(0) = {self.solver.initial_state}",
            f"Time interval: [{self.solver.t_span[0]}, {self.solver.t_span[1]}]",
            f"Step size: h = {self.solver.h}",
            f"Total steps: {self.solver.n_steps}",
            "",
            "Intensity Matrix Q:",
            "-" * 40,
        ]
        
        for i in range(self.solver.n_states):
            row = "  [" + " ".join([f"{self.solver.Q[i,j]:8.4f}" for j in range(self.solver.n_states)]) + "]"
            lines.append(row)
        
        lines.append("")
        return "\n".join(lines)
    
    def _generate_results(self):
        """Generate results section."""
        lines = [
            "3. NUMERICAL SOLUTION RESULTS",
            "=" * 80,
            "",
            "3.1 Solution at Control Points",
            "-" * 40,
            "",
        ]
        
        # Control time points
        control_times = [0, 5, 10, 20, 30]
        control_times = [t for t in control_times if t <= self.solver.t_span[1]]
        
        # Header
        header = f"{'Time':>8} |"
        for i in range(self.solver.n_states):
            header += f" {'P_' + str(i+1) + '(t)':>12} |"
        lines.append(header)
        lines.append("-" * (10 + 15 * self.solver.n_states))
        
        # Values
        for t in control_times:
            y = self.solver.evaluate(t)
            row = f"{t:>8.2f} |"
            for i in range(self.solver.n_states):
                row += f" {y[i]:>12.6f} |"
            lines.append(row)
        
        lines.append("")
        
        # Final state
        lines.append("3.2 Final Probabilities")
        lines.append("-" * 40)
        
        y_final = self.solver.P_values[:, -1]
        for i in range(self.solver.n_states):
            lines.append(f"  P_{i+1}(t_final) = {y_final[i]:.6f}")
        
        lines.append(f"\n  Sum check: {np.sum(y_final):.6f} (should be ≈ 1.0)")
        lines.append("")
        
        # Execution info
        lines.append("3.3 Computation Statistics")
        lines.append("-" * 40)
        lines.append(f"  Execution time: {self.solver.execution_time*1000:.2f} ms")
        lines.append(f"  Steps computed: {self.solver.n_steps}")
        # Calculate max probability deviation
        if hasattr(self.solver, 'P_values') and self.solver.P_values is not None:
            prob_sums = np.sum(self.solver.P_values, axis=0)
            max_dev = np.max(np.abs(prob_sums - 1.0))
            lines.append(f"  Probability conservation: max deviation = {max_dev:.2e}")
        else:
            lines.append(f"  Probability conservation: N/A")
        lines.append("")
        
        return "\n".join(lines)
    
    def _generate_error_analysis(self):
        """Generate error analysis section."""
        if self.comparison_results is None:
            return "4. ERROR ANALYSIS\n" + "=" * 80 + "\n\nError analysis skipped (L2 solution not available).\n"
        
        metrics = self.comparison_results['global_metrics']
        max_dev = self.comparison_results['max_deviation']
        
        lines = [
            "4. ERROR ANALYSIS (vs L2 Analytical Solution)",
            "=" * 80,
            "",
            "4.1 Global Error Metrics",
            "-" * 40,
            f"  Max absolute error:     {metrics['max_abs_error']:.6e}",
            f"  Mean absolute error:    {metrics['mean_abs_error']:.6e}",
            f"  RMSE:                   {metrics['rmse']:.6e}",
            f"  Max relative error:     {metrics['max_rel_error']:.6e}",
            f"  Mean relative error:    {metrics['mean_rel_error']:.6e}",
            "",
            "4.2 Maximum Deviation Details",
            "-" * 40,
            f"  State:                  S_{max_dev['state']}",
            f"  Time:                   t = {max_dev['time']:.4f}",
            f"  Absolute deviation:     {max_dev['value']:.6e}",
            f"  Relative deviation:     {max_dev['relative_value']:.6e}",
            f"  Interval of max dev:    [{max_dev['interval'][0]:.4f}, {max_dev['interval'][1]:.4f}]",
            "",
            "4.3 Per-State Error Summary",
            "-" * 40,
        ]
        
        header = f"{'State':>8} | {'Max Abs':>14} | {'Mean Abs':>14} | {'RMSE':>14} | {'Max Rel':>12}"
        lines.append(header)
        lines.append("-" * 80)
        
        for sm in self.comparison_results['state_metrics']:
            row = (f"  P_{sm['state']:<4} | {sm['max_abs_error']:>14.6e} | "
                   f"{sm['mean_abs_error']:>14.6e} | {sm['rmse']:>14.6e} | "
                   f"{sm['max_rel_error']:>12.6e}")
            lines.append(row)
        
        lines.append("")
        return "\n".join(lines)
    
    def _generate_conclusions(self):
        """Generate conclusions section."""
        lines = [
            "6. CONCLUSIONS",
            "=" * 80,
            "",
            "6.1 Method Performance",
            "-" * 40,
        ]
        
        # Get metrics for conclusion
        if self.comparison_results:
            max_err = self.comparison_results['global_metrics']['max_abs_error']
            rel_err = self.comparison_results['global_metrics']['max_rel_error']
            
            lines.append(f"  - Maximum absolute error: {max_err:.6e}")
            lines.append(f"  - Maximum relative error: {rel_err:.6e}")
            
            if max_err < 1e-4:
                lines.append("  - Accuracy: Excellent (error < 0.01%)")
            elif max_err < 1e-3:
                lines.append("  - Accuracy: Very good (error < 0.1%)")
            elif max_err < 1e-2:
                lines.append("  - Accuracy: Good (error < 1%)")
            else:
                lines.append("  - Accuracy: Moderate (consider smaller step)")
        
        lines.append("")
        lines.append("6.2 Recommendations")
        lines.append("-" * 40)
        
        if self.comparison_results:
            max_err = self.comparison_results['global_metrics']['max_abs_error']
            current_h = self.solver.h
            
            if max_err > 1e-3:
                rec_h = current_h / 2
                lines.append(f"  - For better accuracy, consider using step h = {rec_h:.4f} or smaller")
            else:
                lines.append(f"  - Current step h = {current_h:.4f} provides acceptable accuracy")
            
            lines.append(f"  - Execution time: {self.solver.execution_time*1000:.2f} ms")
        
        lines.append("")
        lines.append("6.3 Control Questions Summary")
        lines.append("-" * 40)
        lines.append("  1. Convergence: The numerical solution approaches exact solution as h→0")
        lines.append("  2. Complexity growth: Linear O(n) with number of variables per step")
        lines.append("  3. Order of accuracy: p=2 for Modified Euler (error ~ O(h²))")
        lines.append("  4. Method comparison:")
        lines.append("     - Euler: 1 eval/step, O(h), simplest")
        lines.append("     - Modified Euler: 2 evals/step, O(h²), good balance")
        lines.append("     - Runge-Kutta 4: 4 evals/step, O(h⁴), highest accuracy")
        lines.append("  5. System properties affecting method choice:")
        lines.append("     - System size, required accuracy, stiffness, connectivity")
        lines.append("")
        
        return "\n".join(lines)
    
    def _generate_footer(self):
        """Generate report footer."""
        lines = [
            "=" * 80,
            "END OF REPORT",
            "=" * 80,
            "",
            "Generated files:",
            "  - L3_results.txt: This report",
            "  - L3_probabilities.png: Probability evolution plot",
            "",
        ]
        return "\n".join(lines)


def generate_report(solver, comparison_results=None):
    """
    Convenience function to generate report.
    
    Args:
        solver: ModifiedEulerSolver instance
        comparison_results: Results from L3L2Comparator.compare()
        
    Returns:
        str: Report text
    """
    generator = L3ReportGenerator(solver, comparison_results)
    return generator.generate()


if __name__ == "__main__":
    print("L3 Report Generator")
    print("=" * 70)
    print("Use this module through L3_report.py main workflow.")

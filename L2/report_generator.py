"""
Report generator for L2 results.
Creates detailed text report with analytical formulas and comparison.

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-16
"""

import os
import numpy as np
from datetime import datetime


class L2ReportGenerator:
    """
    Generates detailed report for L2 analytical solution.
    """
    
    def __init__(self, solver, comparison_results=None, output_path=None):
        """
        Initialize report generator.
        
        Args:
            solver (OperatorSolver): Solved solver instance
            comparison_results (dict): Results from comparison with L1
            output_path (str): Path to save report
        """
        self.solver = solver
        self.comparison_results = comparison_results
        
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(script_dir)
            output_dir = os.path.join(root_dir, 'Output')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'L2_results.txt')
        
        self.output_path = output_path
    
    def generate(self):
        """
        Generate and save report.
        
        Returns:
            str: Report content
        """
        lines = []
        
        # Header
        lines.extend(self._generate_header())
        lines.append("")
        
        # System description
        lines.extend(self._generate_system_description())
        lines.append("")
        
        # Operator method steps
        lines.extend(self._generate_operator_method())
        lines.append("")
        
        # Analytical solution
        lines.extend(self._generate_analytical_solution())
        lines.append("")
        
        # Comparison with L1
        if self.comparison_results:
            lines.extend(self._generate_comparison())
            lines.append("")
        
        # Steady-state analysis
        lines.extend(self._generate_steady_state())
        lines.append("")
        
        # Footer
        lines.extend(self._generate_footer())
        
        report = '\n'.join(lines)
        
        # Save to file
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report saved: {self.output_path}")
        return report
    
    def _generate_header(self):
        """Generate report header."""
        return [
            "=" * 80,
            "LAB 2: OPERATOR METHOD FOR KOLMOGOROV EQUATIONS",
            "Analytical Solution using Laplace Transform",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80
        ]
    
    def _generate_system_description(self):
        """Generate system description section."""
        lines = [
            "1. SYSTEM DESCRIPTION",
            "-" * 80,
            f"Number of states: {self.solver.n_states}",
            f"Initial state: P_{np.argmax(self.solver.initial_state) + 1}(0) = 1",
            f"Absorbing states: {[s + 1 for s in self.solver.absorbing_states]}",
            f"Transient states: {[i + 1 for i in range(self.solver.n_states) if i not in self.solver.absorbing_states]}",
            "",
            "Intensity Matrix Q:",
        ]
        
        # Matrix Q
        for i in range(self.solver.n_states):
            row = "  ["
            for j in range(self.solver.n_states):
                val = self.solver.Q[i, j]
                if abs(val) < 1e-10:
                    row += "     0  "
                else:
                    row += f"{val:7.4f} "
            row += "]"
            lines.append(row)
        
        # Original equations
        lines.extend([
            "",
            "Original Differential Equations (Kolmogorov Forward):",
        ])
        
        for i in range(self.solver.n_states):
            eq = f"  dP_{i+1}/dt = "
            terms = []
            # Positive terms (incoming)
            for j in range(self.solver.n_states):
                if i != j and self.solver.Q[j, i] > 0:
                    terms.append(f"{self.solver.Q[j, i]:.2f}·P_{j+1}")
            # Negative term (outgoing)
            if self.solver.Q[i, i] < 0:
                terms.append(f"{self.solver.Q[i, i]:.2f}·P_{i+1}")
            if not terms:
                terms.append("0")
            eq += " + ".join(terms).replace("+ -", "- ")
            lines.append(eq)
        
        return lines
    
    def _generate_operator_method(self):
        """Generate operator method explanation."""
        lines = [
            "2. OPERATOR METHOD (LAPLACE TRANSFORM)",
            "-" * 80,
            "Applying Laplace transform to the system:",
            "  L{dP_i/dt} = p·P_i(p) - P_i(0)",
            "",
            "Operator system: (pI - Q^T)·P(p) = P(0)",
            "",
            "where:",
            "  p - complex frequency variable",
            "  I - identity matrix",
            "  Q^T - transposed intensity matrix",
            "  P(p) - vector of Laplace transforms",
            "  P(0) - initial conditions vector",
            "",
            "Solution in p-domain:",
            "  P(p) = (pI - Q^T)^(-1) · P(0)",
        ]
        
        if self.solver.eigenvalues is not None and len(self.solver.eigenvalues) > 0:
            lines.extend([
                "",
                "Characteristic equation roots (eigenvalues of Q):",
            ])
            eigenvalues_list = list(self.solver.eigenvalues)
            for i, ev in enumerate(eigenvalues_list[:10]):  # Limit output
                lines.append(f"  λ_{i+1} = {ev}")
            if len(eigenvalues_list) > 10:
                lines.append(f"  ... and {len(eigenvalues_list) - 10} more")
        
        return lines
    
    def _generate_analytical_solution(self):
        """Generate analytical solution section."""
        lines = [
            "3. ANALYTICAL SOLUTION",
            "-" * 80,
            "Solution P(t) obtained by inverse Laplace transform:",
            "",
        ]
        
        formulas = self.solver.get_analytical_formulas()
        for formula in formulas:
            lines.append(formula)
            lines.append("")
        
        return lines
    
    def _generate_comparison(self):
        """Generate comparison section."""
        lines = [
            "4. COMPARISON WITH L1 (NUMERICAL SOLUTION)",
            "-" * 80,
            "Comparing analytical (L2) vs numerical (L1) solutions:",
            "",
            "Global Error Metrics:",
            f"  Maximum absolute error:  {self.comparison_results['global_metrics']['max_abs_error']:.6e}",
            f"  Mean absolute error:     {self.comparison_results['global_metrics']['mean_abs_error']:.6e}",
            f"  Maximum relative error:  {self.comparison_results['global_metrics']['max_rel_error']:.6e}",
            f"  Root mean square error:  {self.comparison_results['global_metrics']['rmse']:.6e}",
            "",
            "Per-State Error Metrics:",
            f"{'State':<10}{'Max Abs':<15}{'Mean Abs':<15}{'Max Rel':<15}{'RMSE':<15}",
            "-" * 70,
        ]
        
        for m in self.comparison_results['state_metrics']:
            lines.append(
                f"P_{m['state']:<9}{m['max_abs_error']:<15.6e}{m['mean_abs_error']:<15.6e}"
                f"{m['max_rel_error']:<15.6e}{m['rmse']:<15.6e}"
            )
        
        lines.extend([
            "",
            "Sample Values at Key Time Points:",
            f"{'Time':<10}{'State':<10}{'L2 (Anal)':<15}{'L1 (Num)':<15}{'Error':<15}",
            "-" * 65,
        ])
        
        # Sample at specific time points
        t = self.comparison_results['t']
        y_l1 = self.comparison_results['y_l1']
        y_l2 = self.comparison_results['y_l2']
        sample_times = [0, 5, 10, 20, 30]
        
        for t_target in sample_times:
            if t_target > t[-1]:
                continue
            idx = np.argmin(np.abs(t - t_target))
            t_actual = t[idx]
            
            for i in range(min(3, self.solver.n_states)):  # Show first 3 states
                lines.append(
                    f"{t_actual:<10.1f}P_{i+1:<9}{y_l2[i, idx]:<15.6f}{y_l1[i, idx]:<15.6f}"
                    f"{abs(y_l2[i, idx] - y_l1[i, idx]):<15.6e}"
                )
            lines.append("")
        
        return lines
    
    def _generate_steady_state(self):
        """Generate steady-state analysis."""
        steady = self.solver.get_steady_state()
        
        lines = [
            "5. STEADY-STATE SOLUTION (t → ∞)",
            "-" * 80,
            "Probabilities at equilibrium:",
            "",
        ]
        
        for i in range(self.solver.n_states):
            state_type = "[absorbing]" if i in self.solver.absorbing_states else "[transient]"
            lines.append(f"  P_{i+1}(∞) = {steady[i]:.6f} {state_type}")
        
        lines.extend([
            "",
            f"Sum of probabilities: {np.sum(steady):.6f}",
        ])
        
        # Absorption probabilities
        if self.solver.absorbing_states:
            lines.extend([
                "",
                "Absorption Probabilities:",
                f"From initial state P_{np.argmax(self.solver.initial_state) + 1}(0) = 1:",
            ])
            for i in self.solver.absorbing_states:
                lines.append(f"  Probability of absorption in state {i+1}: {steady[i]:.6f} ({steady[i]*100:.2f}%)")
        
        return lines
    
    def _generate_footer(self):
        """Generate report footer."""
        return [
            "",
            "=" * 80,
            "END OF REPORT",
            "Generated by MathModel L2 - Operator Method Solver",
            "=" * 80,
        ]


def generate_report(solver, comparison_results=None, output_path=None):
    """
    Convenience function to generate report.
    
    Args:
        solver (OperatorSolver): Solved solver
        comparison_results (dict): Comparison with L1
        output_path (str): Output file path
        
    Returns:
        str: Report content
    """
    generator = L2ReportGenerator(solver, comparison_results, output_path)
    return generator.generate()

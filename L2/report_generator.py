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
        
        # P-domain solution (Cramer's rule)
        lines.extend(self._generate_p_domain_solution())
        lines.append("")
        
        # Partial fraction decomposition
        lines.extend(self._generate_partial_fractions())
        lines.append("")
        
        # Inverse Laplace transform - explicit formulas
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
            "2.1 Applying Laplace transform to the system:",
            "    L{dP_i/dt} = p·P_i(p) - P_i(0)",
            "",
            "2.2 Operator system: (pI - Q^T)·P(p) = P(0)",
            "",
            "    where:",
            "      p - complex frequency variable",
            "      I - identity matrix",
            "      Q^T - transposed intensity matrix",
            "      P(p) - vector of Laplace transforms",
            "      P(0) - initial conditions vector",
            "",
            "2.3 Solution in p-domain (matrix form):",
            "    P(p) = (pI - Q^T)^(-1) · P(0)",
            "",
            "    By Cramer's rule for each component:",
            "    P_i(p) = det(M_i) / det(M)",
            "",
            "    where M = (pI - Q^T), M_i is M with i-th column replaced by P₀",
        ]
        
        # Add matrix M = (pI - Q^T) with symbolic p
        lines.extend([
            "",
            "    Matrix M = (pI - Q^T):",
        ])
        Q_sym = self.solver.Q.T
        for i in range(self.solver.n_states):
            row = "    ["
            for j in range(self.solver.n_states):
                if i == j:
                    # Diagonal: p - Q^T[i,i] = p + |q_ii| (since q_ii < 0)
                    val = -Q_sym[i, j]
                    row += f" p{val:+7.4f} "
                else:
                    # Off-diagonal: -Q^T[i,j]
                    val = -Q_sym[i, j]
                    row += f" {val:7.4f} "
            row += "]"
            lines.append(row)
        
        # Characteristic polynomial
        poly = self.solver.get_characteristic_polynomial()
        if poly is not None:
            lines.extend([
                "",
                "2.4 Characteristic polynomial:",
                "    det(M) = det(pI - Q^T) = (p - λ₁)(p - λ₂)...(p - λₙ)",
            ])
            
            # Format polynomial
            poly_str = "    det(M) = "
            terms = []
            n = len(poly) - 1
            for i, coef in enumerate(poly):
                power = n - i
                if abs(coef) < 1e-10:
                    continue
                if power == 0:
                    term = f"{coef.real:.6f}"
                elif power == 1:
                    term = f"{coef.real:.6f}·p"
                else:
                    term = f"{coef.real:.6f}·p^{power}"
                terms.append(term)
            poly_str += " + ".join(terms).replace("+ -", "- ")
            lines.append(poly_str)
        
        if self.solver.eigenvalues is not None and len(self.solver.eigenvalues) > 0:
            lines.extend([
                "",
                "    Roots of characteristic equation (poles of Laplace transform):",
            ])
            eigenvalues_list = list(self.solver.eigenvalues)
            for i, ev in enumerate(eigenvalues_list):
                if isinstance(ev, (complex, np.complexfloating)):
                    if abs(ev.imag) < 1e-10:
                        lines.append(f"    λ_{i+1} = {ev.real:.6f}")
                    else:
                        lines.append(f"    λ_{i+1} = {ev.real:.6f} {ev.imag:+.6f}i")
                else:
                    lines.append(f"    λ_{i+1} = {ev:.6f}")
        
        return lines
    
    def _generate_p_domain_solution(self):
        """Generate p-domain solution section (4.2 equivalent)."""
        lines = [
            "3. SOLUTION IN P-DOMAIN (CRAMER'S RULE)",
            "-" * 80,
            "3.1 Explicit formulas for P_i(p):",
            "",
            "    Using Cramer's rule for the system (pI - Q^T)·P(p) = P₀:",
            "    P_i(p) = det(M_i) / det(M)",
            "",
            "    where M_i is the matrix M with its i-th column replaced by P₀.",
            "",
        ]
        
        # Add explicit formulas from Laplace domain
        p_formulas = self.solver.get_p_domain_formulas()
        lines.extend(p_formulas)
        
        # Add Cramer's rule numerators
        lines.extend([
            "",
            "3.2 Cramer's rule - explicit numerators det(M_i):",
            "",
            "    For P_i(p) = det(M_i) / det(M):",
            "",
        ])
        
        numerators = self.solver.get_numerators_cramer()
        if numerators is not None:
            # Get denominator coefficients
            poly = self.solver.get_characteristic_polynomial()
            
            for i, coeffs in enumerate(numerators):
                degree = len(coeffs) - 1
                # Format numerator polynomial
                terms = []
                for j, coef in enumerate(coeffs):
                    power = degree - j
                    if abs(coef) > 1e-8:
                        sign = "+" if coef >= 0 else "-"
                        abs_coef = abs(coef)
                        if power == 0:
                            term = f"{abs_coef:.4f}"
                        elif power == 1:
                            term = f"{abs_coef:.4f}p"
                        else:
                            term = f"{abs_coef:.4f}p^{power}"
                        
                        if not terms and sign == "+":
                            terms.append(term)
                        else:
                            terms.append(f"{sign} {term}")
                
                num_formula = " ".join(terms) if terms else "0"
                lines.append(f"    det(M_{i+1}) = {num_formula}")
            
            # Now add full P_i(p) formulas
            lines.extend([
                "",
                "3.3 Explicit formulas P_i(p) = det(M_i) / det(M):",
                "",
            ])
            
            # Format denominator once
            den_terms = []
            den_degree = len(poly) - 1
            for j, coef in enumerate(poly):
                power = den_degree - j
                if abs(coef) > 1e-8:
                    sign = "+" if coef >= 0 else "-"
                    abs_coef = abs(coef)
                    if power == 0:
                        term = f"{abs_coef:.4f}"
                    elif power == 1:
                        term = f"{abs_coef:.4f}p"
                    else:
                        term = f"{abs_coef:.4f}p^{power}"
                    
                    if not den_terms and sign == "+":
                        den_terms.append(term)
                    else:
                        den_terms.append(f"{sign} {term}")
            den_formula = " ".join(den_terms) if den_terms else "0"
            
            for i, coeffs in enumerate(numerators):
                degree = len(coeffs) - 1
                # Format numerator
                num_terms = []
                for j, coef in enumerate(coeffs):
                    power = degree - j
                    if abs(coef) > 1e-8:
                        sign = "+" if coef >= 0 else "-"
                        abs_coef = abs(coef)
                        if power == 0:
                            term = f"{abs_coef:.4f}"
                        elif power == 1:
                            term = f"{abs_coef:.4f}p"
                        else:
                            term = f"{abs_coef:.4f}p^{power}"
                        
                        if not num_terms and sign == "+":
                            num_terms.append(term)
                        else:
                            num_terms.append(f"{sign} {term}")
                num_formula = " ".join(num_terms) if num_terms else "0"
                
                lines.append(f"    P_{i+1}(p) = ({num_formula}) / ({den_formula})")
        else:
            lines.append("    [Numerical computation not available]")
        
        return lines
    
    def _generate_partial_fractions(self):
        """Generate partial fraction decomposition section (4.3 equivalent)."""
        lines = [
            "",
            "4. PARTIAL FRACTION DECOMPOSITION",
            "-" * 80,
            "4.1 Method of undetermined coefficients",
            "",
            "    Each P_i(p) is decomposed as:",
            "    P_i(p) = Σ_{k=1}^n A_{ik} / (p - λ_k)",
            "",
            "    Coefficients are found by:",
            "    A_{ik} = lim_{p→λ_k} [(p - λ_k) · P_i(p)]",
            "",
            "    Using spectral decomposition: A_{ik} = c_k · v_{ik}",
            "    where c = V^(-1)·P₀, v_k are eigenvectors of Q^T",
            "",
            "4.2 Coefficient matrix A_{ik}:",
            "",
            "    Note: For complex conjugate pairs (λₖ = α ± iβ), coefficients",
            "    are combined to produce real-valued expressions.",
        ]
        
        # Add coefficient table
        pf_table = self.solver.get_partial_fraction_table()
        if pf_table is not None:
            eigenvalues = pf_table['eigenvalues']
            n = pf_table['n_states']
            
            # Identify real and complex eigenvalue indices
            real_indices = []
            complex_pairs = []  # (idx1, idx2) for conjugate pairs
            
            processed = set()
            for k in range(n):
                if k in processed:
                    continue
                if abs(eigenvalues[k].imag) < 1e-10:
                    real_indices.append(k)
                else:
                    # Find conjugate
                    for m in range(k+1, n):
                        if abs(eigenvalues[m] - eigenvalues[k].conjugate()) < 1e-10:
                            complex_pairs.append((k, m))
                            processed.add(k)
                            processed.add(m)
                            break
            
            # Header - show real eigenvalues and complex pairs
            header = "    State    |"
            for k in real_indices[:5]:  # Limit to first 5 real
                header += f"   A_{k+1:<2}    |"
            for idx1, idx2 in complex_pairs[:2]:  # Limit to first 2 complex pairs
                header += f" Re(A_{idx1+1})| Im(A_{idx1+1})|"
            lines.append(header)
            
            total_cols = len(real_indices[:5]) + 2 * len(complex_pairs[:2])
            lines.append("    " + "-" * (13 + 12 * total_cols))
            
            # Rows - show real coefficients and combined complex
            for i in range(n):
                row = f"    P_{i+1}(p)   |"
                # Real eigenvalues
                for k in real_indices[:5]:
                    A = pf_table['coefficients'][i, k]
                    row += f" {A.real:8.4f} |"
                # Complex pairs - show real and imag parts
                for idx1, idx2 in complex_pairs[:2]:
                    A1 = pf_table['coefficients'][i, idx1]
                    row += f" {A1.real:8.4f} | {A1.imag:8.4f} |"
                lines.append(row)
        
        return lines
    
    def _generate_analytical_solution(self):
        """Generate analytical solution section (4.4 equivalent)."""
        lines = [
            "",
            "5. INVERSE LAPLACE TRANSFORM",
            "-" * 80,
            "5.1 Applying inverse transform to each term:",
            "",
            "    L⁻¹{A/(p-λ)} = A·e^(λ·t)",
            "",
            "5.2 Explicit formulas P_i(t) = Σ_k A_{ik}·e^(λ_k·t):",
            "",
        ]
        
        formulas = self.solver.get_analytical_formulas()
        for formula in formulas:
            lines.append("    " + formula)
            lines.append("")
        
        # Verification
        lines.extend([
            "5.3 Verification of initial conditions:",
            "",
        ])
        t0_values = self.solver.evaluate(np.array([0]))
        for i in range(self.solver.n_states):
            expected = 1.0 if i == np.argmax(self.solver.initial_state) else 0.0
            actual = t0_values[i, 0]
            check = "✓" if abs(actual - expected) < 1e-6 else "✗"
            lines.append(f"    P_{i+1}(0) = {actual:.6f} (expected {expected:.1f}) {check}")
        
        return lines
    
    def _generate_comparison(self):
        """Generate comparison section."""
        lines = [
            "6. COMPARISON WITH L1 (NUMERICAL SOLUTION)",
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
            "7. STEADY-STATE SOLUTION (t → ∞)",
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

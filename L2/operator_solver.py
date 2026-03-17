"""
Operator method (Laplace transform) solver for linear ODE systems.
Solves Kolmogorov equations analytically.

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-16
"""

import os
import numpy as np
import sympy as sp
from sympy import symbols, Matrix, eye, simplify, apart, exp

from .numerical_inverse_laplace import NumericalLaplaceSolver


class OperatorSolver:
    """
    Solver for linear ODE systems using Laplace transform (operator method).
    
    Solves: dP/dt = P·Q, P(0) = P0
    Using:  pP(p) - P0 = P(p)·Q  →  (pI - Q^T)·P(p) = P0
    """
    
    def __init__(self, Q, initial_state, absorbing_states=None):
        """
        Initialize solver.
        
        Args:
            Q (np.ndarray): Intensity matrix (n_states x n_states)
            initial_state (np.ndarray): Initial probability vector
            absorbing_states (list): List of absorbing state indices (0-based)
        """
        self.Q = np.array(Q, dtype=float)
        self.n_states = self.Q.shape[0]
        self.initial_state = np.array(initial_state, dtype=float)
        self.absorbing_states = absorbing_states or []
        
        # Symbolic variables
        self.p = symbols('p', complex=True)
        self.t = symbols('t', real=True, positive=True)
        
        # Results
        self.P_p = None  # Laplace domain solution
        self.P_t = None  # Time domain solution (symbolic)
        self.P_t_func = None  # Lambdified functions
        self.eigenvalues = None
        
    def solve(self):
        """
        Solve the system using operator method.
        
        For small systems (n <= 4): symbolic solution
        For large systems (n > 4): numerical solution with analytical structure
        
        Returns:
            dict: Solution containing 'P_p' (Laplace), 'P_t' (time), 'eigenvalues'
        """
        print("Operator Method Solver")
        print("=" * 50)
        print(f"Solving system of {self.n_states} states...")
        
        if self.n_states <= 4:
            # Symbolic solution for small systems
            print("Using symbolic solution (small system)")
            self._solve_symbolic()
        else:
            # Numerical solution for large systems
            print("Using numerical solution with eigenvalue decomposition (large system)")
            self._solve_numerical()
        
        print("Solution complete!")
        return {
            'P_p': getattr(self, 'P_p', None),
            'P_t': getattr(self, 'P_t', None),
            'eigenvalues': self.eigenvalues,
            'P_t_func': self.P_t_func,
            'method': 'symbolic' if self.n_states <= 4 else 'numerical'
        }
    
    def _solve_symbolic(self):
        """Full symbolic solution for small systems."""
        # Step 1: Build operator system
        self._build_operator_system()
        
        # Step 2: Solve algebraic system
        self._solve_algebraic()
        
        # Step 3: Partial fraction decomposition
        self._decompose_fractions()
        
        # Step 4: Inverse Laplace transform
        self._inverse_transform()
        
        # Step 5: Lambdify
        self._create_functions()
    
    def _solve_numerical(self):
        """
        Analytical-numerical solution using eigenvalue decomposition for large systems.
        
        For systems > 4 states, we compute the analytical structure numerically:
        1. Find eigenvalues λₖ of Q^T (poles of Laplace transform)
        2. Find coefficients Aᵢₖ for partial fraction expansion
        3. Construct explicit formulas Pᵢ(t) = Σ Aᵢₖ·exp(λₖ·t)
        """
        from numpy.linalg import eig
        
        print("\n[Analytical-Numerical Solution via Spectral Decomposition]")
        
        # Step 1: Compute eigenvalues of Q^T (poles of the solution)
        self.eigenvalues, eigenvectors = eig(self.Q.T)
        
        # Step 2: Solve for coefficients: P₀ = V·c => c = V^(-1)·P₀
        self._eigenvectors = eigenvectors
        self._coeffs = np.linalg.inv(eigenvectors) @ self.initial_state
        
        # Step 3: Compute partial fraction coefficients Aᵢₖ
        # Pᵢ(t) = Σⱼ cⱼ·exp(λⱼ·t)·vᵢⱼ = Σⱼ Aᵢⱼ·exp(λⱼ·t)
        # where Aᵢⱼ = cⱼ·vᵢⱼ
        self.partial_fraction_coeffs = np.zeros((self.n_states, self.n_states), dtype=complex)
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.partial_fraction_coeffs[i, j] = self._coeffs[j] * self._eigenvectors[i, j]
        
        # Step 4: Create explicit analytical formulas
        self._build_analytical_formulas()
        
        # Create lambdified functions for numerical evaluation
        self.P_t_func = []
        for i in range(self.n_states):
            def make_func(idx):
                return lambda t: self._evaluate_state(idx, t)
            self.P_t_func.append(make_func(i))
        
        print(f"\nEigenvalues (poles of Laplace transform):")
        for i, ev in enumerate(self.eigenvalues):
            if abs(ev.imag) < 1e-10:
                print(f"  λ_{i+1} = {ev.real:.6f}")
            else:
                print(f"  λ_{i+1} = {ev.real:.6f} {ev.imag:+.6f}i")
    
    def _build_analytical_formulas(self):
        """Build explicit analytical formulas Pᵢ(t) = Σ Aᵢₖ·exp(λₖ·t)."""
        self.P_t_formulas = []
        
        # Track which eigenvalues have been processed (for complex pairs)
        processed = set()
        
        for i in range(self.n_states):
            terms = []
            
            for j in range(self.n_states):
                if j in processed:
                    continue
                    
                A_ij = self.partial_fraction_coeffs[i, j]
                lambda_j = self.eigenvalues[j]
                
                # Skip negligible coefficients
                if abs(A_ij) < 1e-10 and abs(lambda_j) > 1e-10:
                    continue
                
                # Check if this is a complex eigenvalue with conjugate pair
                is_complex = abs(lambda_j.imag) > 1e-10
                
                if is_complex:
                    # Find conjugate pair
                    conj_idx = None
                    for k in range(j+1, self.n_states):
                        if abs(self.eigenvalues[k] - lambda_j.conjugate()) < 1e-10:
                            conj_idx = k
                            break
                    
                    if conj_idx is not None:
                        # Process complex conjugate pair together
                        A_conj = self.partial_fraction_coeffs[i, conj_idx]
                        
                        # Combine: A·exp((a+ib)t) + A*·exp((a-ib)t) = 2·exp(at)·Re[A·exp(ibt)]
                        # = 2·exp(at)·(Re[A]·cos(bt) - Im[A]·sin(bt))
                        alpha = lambda_j.real
                        beta = lambda_j.imag
                        A_real = A_ij.real
                        A_imag = A_ij.imag
                        
                        # Coefficients for cos and sin
                        cos_coef = 2 * A_real
                        sin_coef = -2 * A_imag
                        
                        # Build term: exp(αt)·(A·cos(βt) + B·sin(βt))
                        if abs(cos_coef) > 1e-10 or abs(sin_coef) > 1e-10:
                            exp_part = f"exp({alpha:.6f}·t)"
                            
                            trig_parts = []
                            if abs(cos_coef) > 1e-10:
                                prefix = "+ " if cos_coef >= 0 else "- "
                                if abs(abs(cos_coef) - 1.0) < 1e-10:
                                    trig_parts.append(f"{prefix}cos({abs(beta):.6f}·t)")
                                else:
                                    trig_parts.append(f"{prefix}{abs(cos_coef):.6f}·cos({abs(beta):.6f}·t)")
                            
                            if abs(sin_coef) > 1e-10:
                                prefix = "+ " if sin_coef >= 0 else "- "
                                if abs(abs(sin_coef) - 1.0) < 1e-10:
                                    trig_parts.append(f"{prefix}sin({abs(beta):.6f}·t)")
                                else:
                                    trig_parts.append(f"{prefix}{abs(sin_coef):.6f}·sin({abs(beta):.6f}·t)")
                            
                            if trig_parts:
                                terms.append(f"+ {exp_part}·({' '.join(trig_parts)})")
                        
                        processed.add(j)
                        processed.add(conj_idx)
                        continue
                
                # Real eigenvalue or unpaired complex
                if abs(lambda_j) < 1e-10:
                    # Constant term (steady-state)
                    if abs(A_ij.real) > 1e-10:
                        terms.append(f"{A_ij.real:.6f}")
                else:
                    # Real exponential term
                    exp_arg = f"{lambda_j.real:.6f}·t"
                    coef = A_ij.real
                    
                    if abs(coef) > 1e-10:
                        prefix = "+ " if coef >= 0 else "- "
                        if abs(abs(coef) - 1.0) < 1e-10:
                            terms.append(f"{prefix}exp({exp_arg})")
                        else:
                            terms.append(f"{prefix}{abs(coef):.6f}·exp({exp_arg})")
                
                processed.add(j)
            
            formula = " ".join(terms) if terms else "0"
            self.P_t_formulas.append(formula)
            processed.clear()  # Reset for next state
        
        # Build p-domain formulas (Laplace transform)
        self.P_p_formulas = []
        for i in range(self.n_states):
            terms = []
            for j in range(self.n_states):
                A_ij = self.partial_fraction_coeffs[i, j]
                lambda_j = self.eigenvalues[j]
                
                if abs(A_ij) < 1e-10:
                    continue
                
                # A/(p - λ) form
                if abs(lambda_j) < 1e-10:
                    terms.append(f"{A_ij.real:.6f}/p")
                else:
                    if abs(lambda_j.imag) < 1e-10:
                        denom = f"(p - {lambda_j.real:.6f})"
                    else:
                        denom = f"(p - ({lambda_j:.6f}))"
                    terms.append(f"{A_ij.real:.6f}/{denom}")
            
            formula = " + ".join(terms) if terms else "0"
            self.P_p_formulas.append(formula)
    
    def _evaluate_state(self, state_idx, t):
        """Evaluate state at time t using eigenvalue decomposition."""
        if np.isscalar(t):
            t = np.array([t])
        else:
            t = np.array(t)
        
        result = np.zeros(len(t), dtype=complex)
        for j in range(self.n_states):
            result += self._coeffs[j] * np.exp(self.eigenvalues[j] * t) * self._eigenvectors[state_idx, j]
        
        return np.real(result)
    
    def _build_operator_system(self):
        """Build operator system matrix M = pI - Q^T."""
        print("\nStep 1: Building operator system...")
        
        # Convert Q to symbolic matrix
        Q_sym = Matrix(self.Q)
        Q_T = Q_sym.T
        
        # Build pI - Q^T
        I = eye(self.n_states)
        self.M = self.p * I - Q_T
        self.P0_vec = Matrix(self.initial_state)
        
        # Compute eigenvalues for reference (numeric for large matrices)
        print(f"Characteristic polynomial degree: {self.n_states}")
        try:
            if self.n_states <= 4:
                self.eigenvalues = list(self.M.eigenvals().keys())
                print(f"Eigenvalues: {[str(ev) for ev in self.eigenvalues[:5]]}{'...' if len(self.eigenvalues) > 5 else ''}")
            else:
                # For large matrices, compute numeric eigenvalues
                Q_num = np.array(self.Q, dtype=float)
                from numpy.linalg import eigvals
                ev_numeric = eigvals(Q_num.T)  # Q^T eigenvalues
                self.eigenvalues = [f"{ev:.4f}" for ev in ev_numeric]
                print(f"Eigenvalues (numeric): {self.eigenvalues}")
        except Exception as e:
            print(f"Note: Could not compute symbolic eigenvalues ({e})")
            self.eigenvalues = []
    
    def _solve_algebraic(self):
        """Solve algebraic system M·P(p) = P0."""
        print("\nStep 2: Solving algebraic system...")
        
        # Solve using matrix inversion: P(p) = M^(-1) · P0
        try:
            M_inv = self.M.inv()
            self.P_p = M_inv * self.P0_vec
            
            # Simplify
            print("Simplifying...")
            for i in range(self.n_states):
                self.P_p[i] = simplify(self.P_p[i])
                
        except Exception as e:
            print(f"Warning: Matrix inversion failed ({e}), using linear solve...")
            self.P_p = self.M.LUsolve(self.P0_vec)
    
    def _decompose_fractions(self):
        """Perform partial fraction decomposition."""
        print("\nStep 3: Partial fraction decomposition...")
        
        self.P_p_decomp = []
        for i in range(self.n_states):
            # Decompose each component
            decomp = apart(self.P_p[i], self.p)
            self.P_p_decomp.append(decomp)
            print(f"  P_{i+1}(p): {len(str(decomp))} chars")
    
    def _inverse_transform(self):
        """Apply inverse Laplace transform."""
        print("\nStep 4: Inverse Laplace transform...")
        
        self.P_t = []
        for i in range(self.n_states):
            print(f"  Transforming P_{i+1}...", end=' ')
            try:
                # Use partial fraction form for better transformation
                p_t = inverse_laplace_transform(self.P_p_decomp[i], self.p, self.t)
                p_t = simplify(p_t)
                self.P_t.append(p_t)
                print(f"OK ({len(str(p_t))} chars)")
            except Exception as e:
                print(f"Failed: {e}")
                self.P_t.append(None)
    
    def _create_functions(self):
        """Create lambdified functions for numerical evaluation."""
        print("\nStep 5: Creating numerical functions...")
        
        self.P_t_func = []
        for i in range(self.n_states):
            if self.P_t[i] is not None:
                try:
                    # Lambdify for numerical evaluation
                    func = sp.lambdify(self.t, self.P_t[i], 'numpy')
                    self.P_t_func.append(func)
                except Exception as e:
                    print(f"Warning: Could not lambdify P_{i+1}: {e}")
                    self.P_t_func.append(None)
            else:
                self.P_t_func.append(None)
    
    def evaluate(self, t_values):
        """
        Evaluate solution at given time points.
        
        Args:
            t_values (np.ndarray): Time points
            
        Returns:
            np.ndarray: Solution array (n_states x len(t_values))
        """
        if self.P_t_func is None:
            raise RuntimeError("Solver not run yet. Call solve() first.")
        
        t_values = np.array(t_values)
        
        # Check if using numerical method (has _eigenvectors)
        if hasattr(self, '_eigenvectors'):
            # Numerical evaluation using eigenvalue decomposition
            result = np.zeros((self.n_states, len(t_values)), dtype=complex)
            for j in range(self.n_states):
                for k in range(self.n_states):
                    result[j, :] += self._coeffs[k] * np.exp(self.eigenvalues[k] * t_values) * self._eigenvectors[j, k]
            result = np.real(result)
        else:
            # Symbolic evaluation
            result = np.zeros((self.n_states, len(t_values)))
            for i in range(self.n_states):
                if self.P_t_func[i] is not None:
                    try:
                        result[i, :] = self.P_t_func[i](t_values)
                    except Exception as e:
                        print(f"Warning: Evaluation failed for P_{i+1}: {e}")
                        result[i, :] = 0
                else:
                    result[i, :] = 0
        
        # Ensure probabilities are non-negative and sum to 1
        result = np.maximum(result, 0)
        sums = np.sum(result, axis=0)
        sums = np.where(sums == 0, 1, sums)  # Avoid division by zero
        result = result / sums
        
        return result
    
    def get_analytical_formulas(self):
        """
        Get analytical formulas P_i(t) as strings.
        
        Returns:
            list: List of formula strings
        """
        formulas = []
        
        if hasattr(self, 'P_t_formulas'):
            # Analytical-numerical method
            formulas.append("Analytical solution via spectral decomposition:")
            formulas.append("P_i(t) = Σ_k A_ik · exp(λ_k · t)")
            formulas.append("")
            formulas.append("Eigenvalues (poles of Laplace transform):")
            for j, ev in enumerate(self.eigenvalues):
                if abs(ev.imag) < 1e-10:
                    formulas.append(f"  λ_{j+1} = {ev.real:.6f}")
                else:
                    formulas.append(f"  λ_{j+1} = {ev.real:.6f} {ev.imag:+.6f}i")
            formulas.append("")
            formulas.append("Explicit formulas:")
            for i in range(self.n_states):
                formulas.append(f"P_{i+1}(t) = {self.P_t_formulas[i]}")
        elif hasattr(self, '_eigenvectors'):
            # Fallback: show structure
            formulas.append("Analytical solution via eigenvalue decomposition:")
            formulas.append("P_i(t) = Σ_k A_ik · exp(λ_k · t)")
            formulas.append("")
            for i in range(self.n_states):
                terms = []
                for j in range(self.n_states):
                    A_ij = self._coeffs[j] * self._eigenvectors[i, j]
                    if abs(A_ij) > 1e-10:
                        if abs(self.eigenvalues[j]) < 1e-10:
                            terms.append(f"{A_ij.real:.6f}")
                        else:
                            terms.append(f"{A_ij.real:.6f}·exp({self.eigenvalues[j].real:.6f}·t)")
                formula = " + ".join(terms) if terms else "0"
                formulas.append(f"P_{i+1}(t) = {formula}")
        else:
            # Symbolic method
            for i in range(self.n_states):
                if self.P_t[i] is not None:
                    formulas.append(f"P_{i+1}(t) = {self.P_t[i]}")
                else:
                    formulas.append(f"P_{i+1}(t) = [not available]")
        
        return formulas
    
    def get_p_domain_formulas(self):
        """
        Get Laplace domain formulas P_i(p) as strings.
        
        Returns:
            list: List of formula strings for P(p)
        """
        formulas = []
        
        if hasattr(self, 'partial_fraction_coeffs'):
            formulas.append("Laplace domain solution (partial fraction expansion):")
            formulas.append("P_i(p) = Σ_k A_ik / (p - λ_k)")
            formulas.append("")
            
            for i in range(self.n_states):
                terms = []
                processed = set()
                
                for j in range(self.n_states):
                    if j in processed:
                        continue
                        
                    A_ij = self.partial_fraction_coeffs[i, j]
                    lambda_j = self.eigenvalues[j]
                    
                    if abs(A_ij) < 1e-10:
                        processed.add(j)
                        continue
                    
                    # Check for complex conjugate pair
                    is_complex = abs(lambda_j.imag) > 1e-10
                    if is_complex:
                        conj_idx = None
                        for k in range(j+1, self.n_states):
                            if abs(self.eigenvalues[k] - lambda_j.conjugate()) < 1e-10:
                                conj_idx = k
                                break
                        
                        if conj_idx is not None:
                            A_conj = self.partial_fraction_coeffs[i, conj_idx]
                            # Combine: (A(p-α) + Bβ) / ((p-α)² + β²)
                            alpha = lambda_j.real
                            beta = abs(lambda_j.imag)
                            A_real = A_ij.real
                            A_imag = A_ij.imag
                            
                            # Real numerator coefficient
                            num_real = 2 * A_real
                            num_imag = -2 * A_imag
                            
                            if abs(lambda_j.imag) < 1e-10:
                                denom = f"(p - {alpha:.6f})"
                            else:
                                if abs(alpha) < 1e-10:
                                    denom = f"(p² + {beta**2:.6f})"
                                else:
                                    denom = f"((p - {alpha:.6f})² + {beta**2:.6f})"
                            
                            prefix = "+ " if num_real >= 0 else "- "
                            terms.append(f"{prefix}{abs(num_real):.6f}·(p - {alpha:.6f})/{denom}")
                            
                            if abs(num_imag) > 1e-10:
                                prefix = "+ " if num_imag >= 0 else "- "
                                terms.append(f"{prefix}{abs(num_imag):.6f}·{beta:.6f}/{denom}")
                            
                            processed.add(j)
                            processed.add(conj_idx)
                            continue
                    
                    # Real case
                    if abs(lambda_j) < 1e-10:
                        prefix = "+ " if A_ij.real >= 0 else "- "
                        terms.append(f"{prefix}{abs(A_ij.real):.6f}/p")
                    else:
                        prefix = "+ " if A_ij.real >= 0 else "- "
                        if abs(lambda_j.real) < 1e-10:
                            denom = "p"
                        else:
                            denom = f"(p - {lambda_j.real:.6f})"
                        terms.append(f"{prefix}{abs(A_ij.real):.6f}/{denom}")
                    
                    processed.add(j)
                
                formula = " ".join(terms) if terms else "0"
                formulas.append(f"P_{i+1}(p) = {formula}")
        else:
            formulas.append("Operator solution: P(p) = (pI - Q^T)^(-1) · P₀")
            formulas.append("Use numerical methods for explicit formulas.")
        
        return formulas
    
    def get_partial_fraction_table(self):
        """
        Get partial fraction coefficients as formatted table.
        
        Returns:
            dict: Coefficients matrix and eigenvalues
        """
        if not hasattr(self, 'partial_fraction_coeffs'):
            return None
        
        return {
            'coefficients': self.partial_fraction_coeffs,
            'eigenvalues': self.eigenvalues,
            'n_states': self.n_states
        }
    
    def get_characteristic_polynomial(self):
        """
        Get coefficients of characteristic polynomial det(pI - Q^T).
        
        Returns:
            np.ndarray: Polynomial coefficients [a_n, a_{n-1}, ..., a_0]
        """
        # Characteristic polynomial coefficients from eigenvalues
        # det(pI - Q^T) = (p - λ₁)(p - λ₂)...(p - λₙ)
        if not hasattr(self, 'eigenvalues'):
            return None
        
        # Use numpy to compute polynomial from roots
        poly_coeffs = np.poly(self.eigenvalues)
        return poly_coeffs
    
    def get_numerators_cramer(self):
        """
        Get numerators det(M_i) for Cramer's rule P_i(p) = det(M_i) / det(M).
        
        Returns:
            list: List of coefficient lists for each numerator polynomial [a_n, ..., a_0]
        """
        try:
            # Use numpy polynomial approach
            # For M = pI - Q^T, we evaluate det at multiple points and fit polynomial
            from numpy.polynomial import polynomial as P
            
            n = self.n_states
            Q_T = self.Q.T
            P0 = self.initial_state
            
            # For each state i, compute det(M_i) at multiple p values
            numerators = []
            
            for state_idx in range(n):
                # Evaluate at multiple points to fit polynomial
                # Degree of numerator is at most n-1 (since one column is constants)
                degree = n - 1
                p_values = np.linspace(0, 10, degree + 1)
                det_values = []
                
                for p_val in p_values:
                    # M = pI - Q^T
                    M = p_val * np.eye(n) - Q_T
                    # Replace state_idx column with P0
                    M_i = M.copy()
                    M_i[:, state_idx] = P0
                    det_values.append(np.linalg.det(M_i))
                
                # Fit polynomial (numpy gives coeffs from low to high degree)
                coeffs_low_to_high = np.polyfit(p_values, det_values, degree)
                # Convert to high-to-low (standard form)
                coeffs = coeffs_low_to_high.tolist()
                
                numerators.append(coeffs)
            
            return numerators
        except Exception as e:
            print(f"Warning: Could not compute numerators ({e})")
            return None
    
    def get_steady_state(self):
        """
        Calculate steady-state probabilities (t → ∞).
        
        Returns:
            np.ndarray: Steady-state probabilities
        """
        steady = np.zeros(self.n_states)
        
        if hasattr(self, '_eigenvectors'):
            # Numerical method: sum terms with eigenvalue ≈ 0
            for j in range(self.n_states):
                if abs(self.eigenvalues[j]) < 1e-10:
                    # This term contributes to steady state
                    for i in range(self.n_states):
                        steady[i] += np.real(self._coeffs[j] * self._eigenvectors[i, j])
        else:
            # Symbolic method
            if self.P_t is None:
                raise RuntimeError("Solver not run yet. Call solve() first.")
            
            for i in range(self.n_states):
                if self.P_t[i] is not None and isinstance(self.P_t[i], sp.Expr):
                    # Take limit as t → ∞
                    limit = sp.limit(self.P_t[i], self.t, sp.oo)
                    steady[i] = float(limit) if limit != sp.oo else 0
                else:
                    steady[i] = 0
        
        # Normalize
        steady = np.maximum(steady, 0)
        total = np.sum(steady)
        if total > 0:
            steady = steady / total
        
        return steady


def solve_from_L1(equations_path=None):
    """
    Convenience function to solve system from L1 export.
    
    Args:
        equations_path (str): Path to L1_equations.txt
        
    Returns:
        OperatorSolver: Configured and solved solver instance
    """
    from .equation_parser import load_from_L1
    
    data = load_from_L1(equations_path)
    solver = OperatorSolver(
        Q=data['Q'],
        initial_state=data['initial_state'],
        absorbing_states=data['absorbing_states']
    )
    solver.solve()
    return solver


if __name__ == "__main__":
    # Test with example system (3 states)
    Q = np.array([[-0.5, 0.3, 0.2],
                  [0.4, -0.7, 0.3],
                  [0.0, 0.0, 0.0]])
    P0 = np.array([1.0, 0.0, 0.0])
    
    solver = OperatorSolver(Q, P0, absorbing_states=[2])
    result = solver.solve()
    
    print("\n" + "=" * 50)
    print("Analytical formulas:")
    for formula in solver.get_analytical_formulas():
        print(formula)
    
    print("\nSteady-state:")
    print(solver.get_steady_state())

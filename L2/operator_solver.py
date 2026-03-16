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
        """Numerical solution using eigenvalue decomposition for large systems."""
        from numpy.linalg import eig
        
        # Compute eigenvalues of Q^T (these are the poles of the solution)
        self.eigenvalues, eigenvectors = eig(self.Q.T)
        
        # Solve for coefficients using eigenvector decomposition
        # P(t) = Σ c_i · exp(λ_i · t) · v_i
        # where c = V^(-1) · P0
        coeffs = np.linalg.inv(eigenvectors) @ self.initial_state
        
        # Store for evaluation
        self._eigenvectors = eigenvectors
        self._coeffs = coeffs
        
        # Create lambdified functions
        self.P_t_func = []
        for i in range(self.n_states):
            def make_func(idx):
                return lambda t: self._evaluate_state(idx, t)
            self.P_t_func.append(make_func(i))
        
        # Store symbolic representation
        self.P_t = [f"Sum of {self.n_states} exponential terms" for _ in range(self.n_states)]
        
        print(f"Eigenvalues (poles): {[f'{ev:.4f}' for ev in self.eigenvalues]}")
    
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
        Get analytical formulas as strings.
        
        Returns:
            list: List of formula strings
        """
        formulas = []
        
        if hasattr(self, '_eigenvectors'):
            # Numerical method: show structure
            formulas.append("Numerical solution via eigenvalue decomposition:")
            formulas.append("P(t) = Σ c_i · exp(λ_i · t) · v_i")
            formulas.append("")
            formulas.append("Eigenvalues (poles of Laplace transform):")
            for j, ev in enumerate(self.eigenvalues):
                formulas.append(f"  λ_{j+1} = {ev:.6f}")
            formulas.append("")
            for i in range(self.n_states):
                terms = []
                for j in range(self.n_states):
                    coef = self._coeffs[j] * self._eigenvectors[i, j]
                    if abs(coef) > 1e-10:
                        if abs(self.eigenvalues[j]) < 1e-10:
                            terms.append(f"{coef:.4f}")
                        else:
                            terms.append(f"{coef:.4f}·exp({self.eigenvalues[j]:.4f}·t)")
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

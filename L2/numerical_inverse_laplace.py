"""
Numerical inverse Laplace transform algorithms.
Used for large systems where symbolic inversion is impossible.

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-16
"""

import numpy as np
from numpy.linalg import eig, inv


def numerical_inverse_laplace_talbot(F_p, t, M=32):
    """
    Talbot method for numerical inverse Laplace transform.
    
    Computes f(t) from F(p) using contour integration.
    
    Args:
        F_p: Function F(p) that takes complex p and returns complex value
        t: Time point (scalar or array)
        M: Number of terms in approximation
        
    Returns:
        f(t): Approximation of inverse transform at time t
    """
    if np.isscalar(t):
        t = np.array([t])
    else:
        t = np.array(t)
    
    result = np.zeros(len(t), dtype=complex)
    
    for k, tk in enumerate(t):
        if tk == 0:
            result[k] = 0
            continue
            
        # Talbot contour parameters
        r = 2 * M / (5 * tk)
        
        s = 0
        for m in range(M):
            # Contour point
            theta = m * np.pi / M
            p = r * theta * (1 / np.tan(theta) + 1j)
            dp = r * (1 / np.tan(theta) - theta / np.sin(theta)**2 + 1j)
            
            # Weight
            if m == 0:
                weight = 0.5
            else:
                weight = 1.0
                
            s += weight * np.exp(tk * p) * F_p(p) * dp
        
        result[k] = s / (2j * np.pi)
    
    return np.real(result)


def matrix_exponential_solution(Q, P0, t):
    """
    Compute solution P(t) = exp(Q^T * t) · P0 using matrix exponential.
    
    This is the formal analytical solution of dP/dt = P·Q.
    
    Args:
        Q: Intensity matrix (n x n)
        P0: Initial state vector (n,)
        t: Time points (array)
        
    Returns:
        P: Solution array (n x len(t))
    """
    from scipy.linalg import expm
    
    n = len(P0)
    t = np.array(t)
    P = np.zeros((n, len(t)))
    
    for i, ti in enumerate(t):
        # P(t) = exp(Q^T * t) · P0
        expQt = expm(Q.T * ti)
        P[:, i] = expQt @ P0
    
    return P


def eigenvalue_decomposition_solution(Q, P0, t):
    """
    Compute solution using eigenvalue decomposition.
    
    P(t) = Σ c_i · exp(λ_i · t) · v_i
    where λ_i, v_i are eigenvalues/eigenvectors of Q^T.
    
    This shows the structure expected from partial fraction decomposition.
    
    Args:
        Q: Intensity matrix (n x n)
        P0: Initial state vector (n,)
        t: Time points (array)
        
    Returns:
        P: Solution array (n x len(t))
        coeffs: Coefficients for each eigenvalue
        eigenvalues: Eigenvalues of Q^T
    """
    n = len(P0)
    t = np.array(t)
    
    # Eigenvalue decomposition of Q^T
    eigenvalues, eigenvectors = eig(Q.T)
    
    # Solve for coefficients: P0 = Σ c_i * v_i
    # c = V^(-1) · P0
    coeffs = inv(eigenvectors) @ P0
    
    # P(t) = Σ c_i · exp(λ_i · t) · v_i
    P = np.zeros((n, len(t)))
    for i, ti in enumerate(t):
        for j in range(n):
            P[:, i] += coeffs[j] * np.exp(eigenvalues[j] * ti) * eigenvectors[:, j]
    
    return P, coeffs, eigenvalues


def partial_fraction_numeric(Q, P0):
    """
    Compute partial fraction coefficients numerically using residues.
    
    For each element X_ij(p) of (pI - Q^T)^(-1), find poles and residues.
    
    Returns:
        poles: List of poles (eigenvalues)
        residues: Residue matrices for each pole
    """
    n = len(P0)
    
    # Eigenvalues of Q^T are the poles
    eigenvalues, _ = eig(Q.T)
    
    # Compute residues at each pole
    residues = []
    for ev in eigenvalues:
        # Residue at p = ev is lim_{p→ev} (p - ev) * (pI - Q^T)^(-1)
        # = projection operator onto eigenvector
        pI_Q = ev * np.eye(n) - Q.T
        # Use pseudo-inverse for numerical stability
        res = np.linalg.pinv(pI_Q)
        residues.append(res)
    
    return eigenvalues, residues


class NumericalLaplaceSolver:
    """
    Solver using numerical inverse Laplace transform for large systems.
    """
    
    def __init__(self, Q, initial_state):
        self.Q = np.array(Q, dtype=float)
        self.n_states = self.Q.shape[0]
        self.initial_state = np.array(initial_state, dtype=float)
        self.eigenvalues = None
        self.coeffs = None
        
    def solve(self, method='eigenvalue'):
        """
        Solve the system numerically.
        
        Args:
            method: 'eigenvalue', 'matrix_exp', or 'talbot'
            
        Returns:
            Solution object with evaluate method
        """
        if method == 'eigenvalue':
            return self._solve_eigenvalue()
        elif method == 'matrix_exp':
            return self._solve_matrix_exp()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _solve_eigenvalue(self):
        """Solve using eigenvalue decomposition."""
        self.eigenvalues, _ = eig(self.Q.T)
        self.coeffs = None  # Computed per evaluation
        self.method = 'eigenvalue'
        return self
    
    def _solve_matrix_exp(self):
        """Solve using matrix exponential."""
        self.method = 'matrix_exp'
        return self
    
    def evaluate(self, t):
        """Evaluate solution at time points t."""
        if self.method == 'eigenvalue':
            P, _, _ = eigenvalue_decomposition_solution(
                self.Q, self.initial_state, t
            )
        elif self.method == 'matrix_exp':
            P = matrix_exponential_solution(
                self.Q, self.initial_state, t
            )
        else:
            raise RuntimeError("Solver not run yet")
        
        # Ensure probabilities are valid
        P = np.real(P)  # Remove small imaginary parts from numerical errors
        P = np.maximum(P, 0)
        P = P / np.sum(P, axis=0)
        
        return P
    
    def get_analytical_form(self):
        """
        Get analytical formula as string.
        
        Returns:
            List of formula strings showing structure
        """
        if self.eigenvalues is None:
            self.eigenvalues, _ = eig(self.Q.T)
        
        formulas = []
        for i in range(self.n_states):
            terms = []
            for j, ev in enumerate(self.eigenvalues):
                if abs(ev) < 1e-10:
                    terms.append(f"c_{j}")
                else:
                    terms.append(f"c_{j}·exp({np.real(ev):.4f}·t)")
            formulas.append(f"P_{i+1}(t) = {' + '.join(terms)}")
        
        return formulas


if __name__ == "__main__":
    # Test with simple system
    Q = np.array([[-1, 1], [0, 0]], dtype=float)
    P0 = np.array([1, 0], dtype=float)
    
    solver = NumericalLaplaceSolver(Q, P0)
    solver.solve(method='eigenvalue')
    
    t = np.linspace(0, 5, 100)
    P = solver.evaluate(t)
    
    print("Numerical solution test:")
    print(f"P(t=0) = {P[:, 0]}")
    print(f"P(t=5) = {P[:, -1]}")
    
    formulas = solver.get_analytical_form()
    print("\nAnalytical form:")
    for f in formulas:
        print(f)

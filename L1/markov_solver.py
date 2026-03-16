"""
Solver for Kolmogorov Forward Equations.
Validates intensity matrix and computes state probabilities over time.

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-16
"""

import json
import os
from datetime import datetime
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class MarkovChainSolver:
    """
    Solver for continuous-time Markov chain with absorbing states.
    
    Attributes:
        n_states (int): Number of states
        Q (np.ndarray): Infinitesimal generator matrix (n_states x n_states)
        initial_state (np.ndarray): Initial probability distribution
        absorbing_states (list): List of absorbing state indices (0-based)
    """
    
    def __init__(self, config=None):
        """
        Initialize Markov chain solver from config.
        
        Args:
            config (dict): Configuration dictionary for L1. If None, loads from root config.json
        """
        if config is None:
            config = self._load_config()
        
        self.n_states = config['n_states']
        
        # Create initial state vector
        self.initial_state = np.zeros(self.n_states)
        initial_idx = config['initial_state']
        if initial_idx < 0 or initial_idx >= self.n_states:
            initial_idx = 0
        self.initial_state[initial_idx] = 1.0
        
        # Build intensity matrix from transitions
        self.Q = self._build_intensity_matrix(config['transitions'])
        
        # Detect absorbing states dynamically (rows with all zeros)
        self.absorbing_states = self._detect_absorbing_states()
        
    def _load_config(self):
        """
        Load L1 configuration from root config.json.
        
        Returns:
            dict: L1 configuration section
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        config_path = os.path.join(root_dir, 'config.json')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = json.load(f)
        
        return full_config['L1']
    
    def _build_intensity_matrix(self, transitions):
        """
        Build infinitesimal generator matrix Q where q_ij = lambda_ij (i≠j)
        and q_ii = -sum(lambda_ij for all j≠i).
        
        Args:
            transitions (list): List of dicts with 'from', 'to', 'rate' keys (1-based indexing)
        
        Returns:
            np.ndarray: n_states x n_states generator matrix
        """
        n = self.n_states
        Q = np.zeros((n, n))
        
        for tr in transitions:
            i = tr['from'] - 1
            j = tr['to'] - 1
            rate = tr['rate']
            Q[i, j] = rate
        
        for i in range(n):
            Q[i, i] = -np.sum(Q[i, :])
            
        return Q
    
    def _detect_absorbing_states(self):
        """
        Detect absorbing states (rows with all zero elements in Q).
        
        Returns:
            list: Indices of absorbing states (0-based)
        """
        absorbing = []
        for i in range(self.n_states):
            if np.allclose(self.Q[i, :], 0):
                absorbing.append(i)
        return absorbing
    
    def validate_matrix(self):
        """
        Validate that Q is a proper infinitesimal generator:
        1. Row sums approximately zero
        2. Off-diagonal elements non-negative
        3. Diagonal elements non-positive
        
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If validation fails
        """
        row_sums = np.sum(self.Q, axis=1)
        if not np.allclose(row_sums, 0, atol=1e-10):
            raise ValueError(f"Row sums not zero: {row_sums}")
        
        off_diag = self.Q.copy()
        np.fill_diagonal(off_diag, 0)
        if np.any(off_diag < 0):
            raise ValueError("Negative off-diagonal elements found")
        
        diag = np.diag(self.Q)
        if np.any(diag > 0):
            raise ValueError("Positive diagonal elements found")
            
        print("Intensity matrix Q is valid")
        if self.absorbing_states:
            print("Absorbing states detected:", [s + 1 for s in self.absorbing_states])
        return True
    
    def kolmogorov_equations(self, t, P):
        """
        Right-hand side of dP/dt = P · Q (Kolmogorov forward equations).
        
        Args:
            t (float): Current time
            P (np.ndarray): Current probability vector (1xn_states)
            
        Returns:
            np.ndarray: Derivative dP/dt
        """
        return P @ self.Q
    
    def solve(self, t_span=(0, 20), t_points=200):
        """
        Solve system of ODEs for state probabilities.
        
        Args:
            t_span (tuple): Time interval (start, end)
            t_points (int): Number of time points for evaluation
            
        Returns:
            OdeResult: SciPy solution object with attributes t (times) and y (probabilities)
        """
        t_eval = np.linspace(t_span[0], t_span[1], t_points)
        
        solution = solve_ivp(
            fun=self.kolmogorov_equations,
            t_span=t_span,
            y0=self.initial_state,
            t_eval=t_eval,
            method='RK45',
            dense_output=True
        )
        
        prob_sums = np.sum(solution.y, axis=0)
        deviation = np.max(np.abs(prob_sums - 1.0))
        print(f"Probability conservation check: max deviation from 1.0 = {deviation:.2e}")
        
        return solution
    
    def get_differential_equations_text(self):
        """
        Generate formatted text of differential equations.
        
        Returns:
            str: LaTeX-ready equations
        """
        lines = []
        lines.append("System generated automatically based on intensity matrix Q (infinitesimal generator),")
        lines.append("where q_ij = lambda_ij (i≠j), and q_ii = -sum(lambda_ij).")
        lines.append("")
        
        state_names = [f'P_{i+1}' for i in range(self.n_states)]
        
        for i in range(self.n_states):
            is_absorbing = i in self.absorbing_states
            state_type = "absorbing" if is_absorbing else "transient"
            lines.append(f"For S_{i+1} ({state_type}):")
            
            terms_in = []
            terms_out = []
            
            for j in range(self.n_states):
                if i != j and self.Q[j, i] > 0:
                    terms_in.append(f"{self.Q[j, i]:.2f}·{state_names[j]}")
            
            if self.Q[i, i] != 0:
                out_coef = -self.Q[i, i]
                terms_out.append(f"{out_coef:.2f}·{state_names[i]}")
            
            # Build equation
            rhs_parts = []
            if terms_in:
                rhs_parts.append(" + ".join(terms_in))
            if terms_out:
                rhs_parts.append(" - ".join(terms_out))
            
            rhs = " + ".join(rhs_parts) if rhs_parts else "0"
            
            # Detailed form with rates
            if not is_absorbing:
                outgoing_rates = []
                for j in range(self.n_states):
                    if i != j and self.Q[i, j] > 0:
                        outgoing_rates.append(f"{self.Q[i, j]:.2f}")
                if outgoing_rates:
                    lines.append(f"d{state_names[i]}/dt = -({' + '.join(outgoing_rates)}){state_names[i]} + ...")
            
            lines.append(f"d{state_names[i]}/dt = {rhs}")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_results_table(self, solution, time_points=[0, 5, 10, 20, 30]):
        """
        Generate results table for specific time points.
        
        Args:
            solution: OdeResult from solve()
            time_points: List of time points to include
            
        Returns:
            str: Formatted table
        """
        lines = []
        lines.append("Integration Results:")
        lines.append("")
        
        # Header
        header = "Time\t"
        for i in range(self.n_states):
            header += f"P_{i+1}(t)\t"
        lines.append(header)
        lines.append("-" * 80)
        
        # Find closest time points
        for t_target in time_points:
            if t_target > solution.t[-1]:
                continue
            idx = np.argmin(np.abs(solution.t - t_target))
            t_actual = solution.t[idx]
            
            row = f"t={t_target}\t"
            for i in range(self.n_states):
                val = solution.y[i, idx]
                if val < 0.0001 and val > 0:
                    row += f"~0\t"
                else:
                    row += f"{val:.4f}\t"
            lines.append(row)
        
        lines.append("")
        lines.append("Note: ~0 means value less than 0.0001")
        return "\n".join(lines)
    
    def save_results(self, solution, output_file=None):
        """
        Save all results to text file.
        
        Args:
            solution: OdeResult from solve()
            output_file (str): Path to save file. If None, saves to Output/L1_results.txt
        """
        if output_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(script_dir)
            output_dir = os.path.join(root_dir, 'Output')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'L1_results.txt')
        
        lines = []
        lines.append("=" * 80)
        lines.append("LAB 1: MARKOV CHAINS")
        lines.append("Simulation Results")
        lines.append("=" * 80)
        lines.append("")
        
        # System description
        lines.append("1. SYSTEM DESCRIPTION")
        lines.append("-" * 80)
        lines.append(f"Number of states: {self.n_states}")
        lines.append(f"Initial state: S_{np.argmax(self.initial_state) + 1}")
        lines.append(f"Absorbing states: {[s + 1 for s in self.absorbing_states]}")
        lines.append(f"Transient states: {[i + 1 for i in range(self.n_states) if i not in self.absorbing_states]}")
        lines.append("")
        
        # Matrix Q
        lines.append("2. INTENSITY MATRIX Q")
        lines.append("-" * 80)
        lines.append("Q =")
        for i in range(self.n_states):
            row = "  ["
            for j in range(self.n_states):
                if self.Q[i, j] == 0:
                    row += "  0   "
                else:
                    row += f"{self.Q[i, j]:5.2f} "
            row += "]"
            lines.append(row)
        lines.append("")
        
        # Differential equations
        lines.append("3. KOLMOGOROV EQUATIONS SYSTEM")
        lines.append("-" * 80)
        lines.append(self.get_differential_equations_text())
        lines.append("")
        
        # Results table
        lines.append("4. INTEGRATION RESULTS")
        lines.append("-" * 80)
        lines.append(self.generate_results_table(solution))
        lines.append("")
        
        # Final probabilities
        lines.append("5. FINAL PROBABILITIES (t → ∞)")
        lines.append("-" * 80)
        for i in range(self.n_states):
            val = solution.y[i, -1]
            state_type = "[absorbing]" if i in self.absorbing_states else "[transient]"
            lines.append(f"P_{i+1}(∞) ≈ {val:.6f} {state_type}")
        lines.append("")
        
        # Conservation check
        prob_sum = np.sum(solution.y[:, -1])
        lines.append(f"Sum of probabilities: {prob_sum:.6f} (should be ≈ 1.0)")
        lines.append("")
        
        # Absorption probabilities
        lines.append("6. ABSORPTION PROBABILITIES")
        lines.append("-" * 80)
        lines.append(f"From initial state S_{np.argmax(self.initial_state) + 1}:")
        
        # Calculate absorption probabilities
        from scipy.linalg import inv
        transient_indices = [i for i in range(self.n_states) if i not in self.absorbing_states]
        if transient_indices and self.absorbing_states:
            Q_transient = self.Q[np.ix_(transient_indices, transient_indices)]
            R = self.Q[np.ix_(transient_indices, self.absorbing_states)]
            N = inv(-Q_transient)
            B = N @ R
            initial_idx = transient_indices.index(np.argmax(self.initial_state))
            
            for j, abs_idx in enumerate(self.absorbing_states):
                prob = B[initial_idx, j]
                lines.append(f"  Absorption probability in S_{abs_idx + 1}: {prob:.6f}")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("End of report")
        lines.append("=" * 80)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        print(f"Results saved to {output_file}")
        
        # Save numerical solution for L2 comparison
        solution_file = os.path.join(output_dir, 'L1_solution.npy')
        np.save(solution_file, {'t': solution.t, 'y': solution.y})
        print(f"Solution saved for L2: {solution_file}")
    
    def export_for_L2(self, output_path=None):
        """
        Export equations in format suitable for L2 operator method.
        
        Args:
            output_path (str): Path to save export file. If None, saves to Output/L1_equations.txt
        """
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(script_dir)
            output_dir = os.path.join(root_dir, 'Output')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'L1_equations.txt')
        
        lines = [
            "# L1 Export for L2 - Kolmogorov Equations",
            f"# Generated: {datetime.now().isoformat()}",
            f"# States: {self.n_states}",
            f"# Initial state index: {np.argmax(self.initial_state)}",
            f"# Absorbing states: {[s + 1 for s in self.absorbing_states]}",
            ""
        ]
        
        # Add matrix Q
        lines.append("# Matrix Q (intensity matrix):")
        for i in range(self.n_states):
            row = "# " + " ".join([f"{self.Q[i,j]:8.4f}" for j in range(self.n_states)])
            lines.append(row)
        lines.append("")
        
        # Add equations in parsed format
        lines.append("# Equations (format: dP_i/dt=...):")
        for i in range(self.n_states):
            eq = f"dP_{i+1}/dt="
            terms = []
            # Incoming terms (positive)
            for j in range(self.n_states):
                if i != j and self.Q[j, i] > 0:
                    terms.append(f"{self.Q[j,i]:.2f}*P_{j+1}")
            # Outgoing term (negative)
            if self.Q[i, i] < 0:
                terms.append(f"{self.Q[i,i]:.2f}*P_{i+1}")
            # Absorbing state with no incoming
            if not terms:
                terms.append("0.00")
            eq += "+".join(terms).replace("+-", "-")
            lines.append(eq)
        
        lines.append("")
        lines.append("# Initial conditions (P_i(0)):")
        for i in range(self.n_states):
            lines.append(f"P_{i+1}(0)={self.initial_state[i]:.1f}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"Exported for L2: {output_path}")
    
    def plot_probabilities(self, solution, output_file=None):
        """
        Plot evolution of state probabilities over time.
        
        Args:
            solution: OdeResult from solve()
            output_file (str): Path to save figure. If None, saves to Output/L1_probabilities.png
        """
        if output_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(script_dir)
            output_dir = os.path.join(root_dir, 'Output')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'L1_probabilities.png')
        
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_states))
        
        for i in range(self.n_states):
            is_absorbing = i in self.absorbing_states
            label = f"P_{i+1}(t)" if not is_absorbing else f"P_{i+1}(t) [absorbing]"
            plt.plot(solution.t, solution.y[i], 
                    label=label, linewidth=2.5, color=colors[i])
        
        plt.xlabel("Time (t)", fontsize=12)
        plt.ylabel("Probability", fontsize=12)
        plt.title("State Probabilities", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='right', fontsize=10)
        plt.xlim(left=0)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Probability plot saved to {output_file}")
    
    def print_differential_equations(self):
        """
        Print formatted system of differential equations for report.
        Generates LaTeX-ready output.
        """
        print("\nSystem of Kolmogorov Forward Equations:")
        print("=" * 50)
        print(self.get_differential_equations_text())
        print("\nInitial conditions:", self.initial_state)


if __name__ == "__main__":
    solver = MarkovChainSolver()
    solver.validate_matrix()
    solver.print_differential_equations()
    
    sol = solver.solve(t_span=(0, 30), t_points=500)
    solver.plot_probabilities(sol)
    solver.save_results(sol)
    
    # Export for L2
    solver.export_for_L2()
    
    print("\nFinal probabilities (t=30):")
    for i, p in enumerate(sol.y[:, -1]):
        print(f"P_{i+1}(∞) ≈ {p:.6f}")

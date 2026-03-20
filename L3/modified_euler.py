"""
Modified Euler Method (Heun's method) solver for linear ODE systems.

This module implements the predictor-corrector method:
    k1 = h * f(t_n, y_n)
    k2 = h * f(t_n + h, y_n + k1)
    y_{n+1} = y_n + (k1 + k2) / 2

Order of accuracy: O(h^2)

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-19
"""

import numpy as np
import time


class ModifiedEulerSolver:
    """
    Solver for linear ODE systems using Modified Euler method.
    
    Solves the system: dP/dt = P · Q (Kolmogorov forward equations)
    with initial condition P(0) = P0.
    
    This is a predictor-corrector method with 2nd order accuracy.
    """
    
    def __init__(self, Q, initial_state, h=0.01, t_span=(0, 30)):
        """
        Initialize Modified Euler solver.
        
        Args:
            Q (np.ndarray): Intensity matrix (n_states x n_states)
            initial_state (np.ndarray): Initial probability vector
            h (float): Integration step size
            t_span (tuple): Time interval (t_start, t_end)
        """
        self.Q = np.array(Q, dtype=float)
        self.n_states = self.Q.shape[0]
        self.initial_state = np.array(initial_state, dtype=float)
        self.h = h
        self.t_span = t_span
        
        # Results storage
        self.t_values = None
        self.P_values = None
        self.execution_time = None
        self.n_steps = None
        
    def f(self, t, P):
        """
        Right-hand side of ODE system: dP/dt = P · Q
        
        For autonomous system (Q constant), t is not used,
        but kept for interface compatibility.
        
        Args:
            t (float): Current time (unused for autonomous system)
            P (np.ndarray): Current probability vector (n_states,)
            
        Returns:
            np.ndarray: Derivative dP/dt (n_states,)
        """
        return P @ self.Q
    
    def step(self, t, P):
        """
        Single step of Modified Euler method.
        
        Predictor-corrector scheme:
            k1 = h * f(t_n, P_n)
            k2 = h * f(t_n + h, P_n + k1)
            P_{n+1} = P_n + (k1 + k2) / 2
        
        Args:
            t (float): Current time
            P (np.ndarray): Current probability vector
            
        Returns:
            np.ndarray: Probability vector at next time step
        """
        # Predictor step (Euler method)
        k1 = self.h * self.f(t, P)
        
        # Corrector step
        k2 = self.h * self.f(t + self.h, P + k1)
        
        # Combined step
        P_next = P + 0.5 * (k1 + k2)
        
        # Ensure non-negative probabilities
        P_next = np.maximum(P_next, 0.0)
        
        # Normalize to ensure sum = 1 (probability conservation)
        P_sum = np.sum(P_next)
        if P_sum > 0:
            P_next = P_next / P_sum
        
        return P_next
    
    def solve(self, store_steps=None):
        """
        Solve ODE system over full time span.
        
        Args:
            store_steps (int, optional): Number of points to store.
                If None, stores all steps. Useful for reducing memory
                when using very small step sizes.
                
        Returns:
            dict: Solution containing:
                - 't': Time values array
                - 'y': Probability values array (n_states x n_times)
                - 'n_steps': Total number of steps computed
                - 'execution_time': Computation time in seconds
        """
        t_start, t_end = self.t_span
        n_steps = int(np.ceil((t_end - t_start) / self.h)) + 1
        
        # Determine storage strategy
        if store_steps is None or store_steps >= n_steps:
            # Store all steps
            store_indices = np.arange(n_steps)
            n_store = n_steps
        else:
            # Store only selected steps (uniformly distributed)
            store_indices = np.linspace(0, n_steps - 1, store_steps, dtype=int)
            n_store = store_steps
        
        # Initialize storage
        t_values = np.zeros(n_store)
        P_values = np.zeros((self.n_states, n_store))
        
        # Initial conditions
        t = t_start
        P = self.initial_state.copy()
        
        store_idx = 0
        if 0 in store_indices:
            t_values[store_idx] = t
            P_values[:, store_idx] = P
            store_idx += 1
        
        # Integration loop
        start_time = time.perf_counter()
        
        for step in range(1, n_steps):
            # Perform Modified Euler step
            P = self.step(t, P)
            t = t_start + step * self.h
            
            # Store if this step is in store_indices
            if step in store_indices:
                t_values[store_idx] = t
                P_values[:, store_idx] = P
                store_idx += 1
        
        end_time = time.perf_counter()
        
        self.execution_time = end_time - start_time
        self.n_steps = n_steps
        self.t_values = t_values
        self.P_values = P_values
        
        # Check probability conservation
        prob_sums = np.sum(P_values, axis=0)
        max_deviation = np.max(np.abs(prob_sums - 1.0))
        
        return {
            't': t_values,
            'y': P_values,
            'n_steps': n_steps,
            'n_stored': n_store,
            'execution_time': self.execution_time,
            'step_size': self.h,
            'max_prob_deviation': max_deviation
        }
    
    def evaluate(self, t_query):
        """
        Evaluate solution at arbitrary time points using linear interpolation.
        
        Args:
            t_query (np.ndarray or float): Time points to evaluate
            
        Returns:
            np.ndarray: Probability values at query points
                       (n_states,) if scalar input
                       (n_states, n_points) if array input
        """
        if self.t_values is None:
            raise RuntimeError("Solver not run yet. Call solve() first.")
        
        scalar_input = np.isscalar(t_query)
        t_query = np.atleast_1d(t_query)
        
        result = np.zeros((self.n_states, len(t_query)))
        
        for i, t in enumerate(t_query):
            if t <= self.t_values[0]:
                result[:, i] = self.P_values[:, 0]
            elif t >= self.t_values[-1]:
                result[:, i] = self.P_values[:, -1]
            else:
                # Find interval
                idx = np.searchsorted(self.t_values, t) - 1
                idx = max(0, min(idx, len(self.t_values) - 2))
                
                # Linear interpolation
                t0, t1 = self.t_values[idx], self.t_values[idx + 1]
                P0, P1 = self.P_values[:, idx], self.P_values[:, idx + 1]
                
                if t1 > t0:
                    frac = (t - t0) / (t1 - t0)
                    result[:, i] = P0 + frac * (P1 - P0)
                else:
                    result[:, i] = P0
        
        if scalar_input:
            return result[:, 0]
        return result
    
    def get_solution_at_points(self, time_points):
        """
        Get solution at specific time points.
        
        Args:
            time_points (list or np.ndarray): Time points
            
        Returns:
            np.ndarray: Solution values (n_states x n_points)
        """
        return self.evaluate(np.array(time_points))
    
    def get_accuracy_estimate(self, reference_solver=None, t_test=None):
        """
        Estimate accuracy by comparing with reference solution
        or by comparing with solution using smaller step.
        
        Args:
            reference_solver: Reference solution (e.g., from L2)
            t_test (np.ndarray): Test time points
            
        Returns:
            dict: Accuracy metrics
        """
        if t_test is None:
            t_test = np.linspace(self.t_span[0], self.t_span[1], 100)
        
        y_numerical = self.evaluate(t_test)
        
        metrics = {
            'time_points': t_test,
            'numerical': y_numerical,
        }
        
        if reference_solver is not None:
            # Compare with reference (analytical or high-precision numerical)
            if hasattr(reference_solver, 'evaluate'):
                y_reference = reference_solver.evaluate(t_test)
            elif isinstance(reference_solver, dict) and 't' in reference_solver:
                # Interpolate from saved solution
                from numpy import interp
                y_reference = np.zeros((self.n_states, len(t_test)))
                for i in range(self.n_states):
                    y_reference[i, :] = np.interp(
                        t_test, 
                        reference_solver['t'], 
                        reference_solver['y'][i, :]
                    )
            else:
                y_reference = None
            
            if y_reference is not None:
                # Compute errors
                abs_error = np.abs(y_numerical - y_reference)
                rel_error = np.zeros_like(abs_error)
                
                # Relative error (avoid division by zero)
                mask = y_reference > 1e-10
                rel_error[mask] = abs_error[mask] / y_reference[mask]
                
                metrics['reference'] = y_reference
                metrics['absolute_error'] = abs_error
                metrics['relative_error'] = rel_error
                metrics['max_absolute_error'] = np.max(abs_error)
                metrics['mean_absolute_error'] = np.mean(abs_error)
                metrics['max_relative_error'] = np.max(rel_error[mask]) if np.any(mask) else 0
                metrics['rmse'] = np.sqrt(np.mean(abs_error**2))
        
        return metrics


def solve_from_L1(equations_path=None, h=0.01, t_span=(0, 30)):
    """
    Convenience function to solve system from L1 export.
    
    Args:
        equations_path (str): Path to L1_equations.txt
        h (float): Integration step
        t_span (tuple): Time interval
        
    Returns:
        ModifiedEulerSolver: Configured and solved solver instance
    """
    import sys
    import os
    
    # Add parent directory to path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    sys.path.insert(0, root_dir)
    
    from L2.equation_parser import load_from_L1
    
    data = load_from_L1(equations_path)
    solver = ModifiedEulerSolver(
        Q=data['Q'],
        initial_state=data['initial_state'],
        h=h,
        t_span=t_span
    )
    solver.solve()
    return solver


if __name__ == "__main__":
    # Test with simple 3-state system
    Q = np.array([[-0.5, 0.3, 0.2],
                  [0.4, -0.7, 0.3],
                  [0.0, 0.0, 0.0]])
    P0 = np.array([1.0, 0.0, 0.0])
    
    print("Modified Euler Method Test")
    print("=" * 50)
    print(f"System: 3 states, absorbing state S_3")
    print(f"Step size: h = 0.01")
    print(f"Time span: [0, 10]")
    print()
    
    solver = ModifiedEulerSolver(Q, P0, h=0.01, t_span=(0, 10))
    result = solver.solve()
    
    print(f"Steps computed: {result['n_steps']}")
    print(f"Execution time: {result['execution_time']*1000:.2f} ms")
    print(f"Max probability deviation from 1.0: {result['max_prob_deviation']:.2e}")
    print()
    
    print("Final probabilities:")
    for i, p in enumerate(result['y'][:, -1]):
        print(f"  P_{i+1}(10) = {p:.6f}")
    
    print("\nSum check:", np.sum(result['y'][:, -1]))

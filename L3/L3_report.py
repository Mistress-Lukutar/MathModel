"""
Main entry point for L3 - Numerical Solution using Modified Euler Method.

Runs the complete L3 workflow:
1. Load data from L1
2. Solve using Modified Euler method
3. Compare with L2 analytical solution
4. Analyze convergence with different step sizes
5. Generate report and plots

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-19
"""

import os
import sys
import json
import numpy as np

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

from L2.equation_parser import load_from_L1
from L3.modified_euler import ModifiedEulerSolver
from L3.comparison import L3L2Comparator
from L3.report_generator import generate_report


def load_config():
    """Load configuration from root config.json."""
    config_path = os.path.join(root_dir, 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_l2_solution():
    """
    Check if L2 solution exists, prompt user if not.
    
    Returns:
        dict or None: L2 solution data if available
    """
    l2_file = os.path.join(root_dir, 'Output', 'L2_solution.npy')
    
    if os.path.exists(l2_file):
        print(f"Found L2 solution: {l2_file}")
        data = np.load(l2_file, allow_pickle=True).item()
        return data
    
    print("WARNING: L2 analytical solution not found!")
    print(f"Expected: {l2_file}")
    print("\nPlease run Lab 2 first for accurate comparison:")
    print("  run_labs.bat → Select [2] Lab 2")
    print("\nContinuing without L2 comparison (error analysis will be skipped).")
    
    return None


def plot_numerical_solution(solver, l2_solution=None, output_dir=None):
    """
    Plot numerical solution and comparison with L2 analytical.
    
    Args:
        solver: ModifiedEulerSolver instance
        l2_solution: L2 solution data for comparison (optional)
        output_dir: Output directory
        
    Returns:
        str: Path to saved file
    """
    import matplotlib.pyplot as plt
    
    if output_dir is None:
        output_dir = os.path.join(root_dir, 'Output')
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Main plot - probabilities (top subplot)
    ax = axes[0]
    t = solver.t_values
    y = solver.P_values
    n_states = solver.n_states
    colors = plt.cm.tab10(np.linspace(0, 1, n_states))
    
    # Plot L3 numerical solutions (solid lines)
    for i in range(n_states):
        ax.plot(t, y[i, :], '-', color=colors[i], linewidth=2, label=f'P_{i+1}(t)')
    
    # If L2 solution available, overlay as points
    if l2_solution is not None:
        # Interpolate L2 to L3 time points
        y_l2 = np.zeros((n_states, len(t)))
        for i in range(n_states):
            y_l2[i, :] = np.interp(t, l2_solution['t'], l2_solution['y'][i, :])
        
        # Plot L2 as points (sparse to avoid clutter)
        step = max(1, len(t) // 60)
        for i in range(n_states):
            ax.plot(t[::step], y_l2[i, ::step], 'o', color=colors[i], markersize=3,
                    markerfacecolor=colors[i], markeredgecolor='black', markeredgewidth=0.5)
        
        # Legend entry for L2 points
        ax.plot([], [], 'o', color='gray', markersize=3, markerfacecolor='gray',
                markeredgecolor='black', markeredgewidth=0.5, label='L2 analytical (points)')
        
        ax.set_title('L3 (Modified Euler - solid lines) vs L2 (Analytical - points)', fontsize=14)
    else:
        ax.set_title('Numerical Solution - Modified Euler Method', fontsize=14)
    
    ax.set_xlabel("Time (t)", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='right', fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    
    # Bottom subplot: Absolute error vs L2 (like L2_comparison.png)
    ax = axes[1]
    
    if l2_solution is not None:
        # Compute error between L3 and L2
        y_l2_interp = np.zeros((n_states, len(t)))
        for i in range(n_states):
            y_l2_interp[i, :] = np.interp(t, l2_solution['t'], l2_solution['y'][i, :])
        
        abs_error = np.abs(y - y_l2_interp)
        max_err = np.max(abs_error)
        
        # Plot error for each state on log scale
        for i in range(n_states):
            ax.semilogy(t, abs_error[i, :] + 1e-15, color=colors[i], 
                       linewidth=1.5, label=f'Error P_{i+1}')
        
        ax.set_title(f'Absolute Error |L3 - L2| - Max: {max_err:.2e}', fontsize=14)
        ax.set_ylabel('Absolute Error (log scale)', fontsize=12)
    else:
        # No L2 available - show probability conservation error
        prob_sums = np.sum(y, axis=0)
        abs_deviation = np.abs(prob_sums - 1.0)
        max_dev = np.max(abs_deviation)
        
        ax.semilogy(t, abs_deviation + 1e-16, 'b-', linewidth=2)
        ax.axhline(y=max_dev, color='r', linestyle='--', linewidth=1.5,
                   label=f'Max deviation = {max_dev:.2e}')
        ax.set_title('Probability Conservation Error |ΣPᵢ - 1|', fontsize=14)
        ax.set_ylabel('Absolute Error (log scale)', fontsize=12)
    
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'L3_probabilities.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved numerical solution plot: {output_file}")
    return output_file


def main():
    """Main L3 workflow."""
    print("=" * 80)
    print("LAB 3: NUMERICAL SOLUTION OF LINEAR DIFFERENTIAL EQUATIONS")
    print("Variant 8: Modified Euler Method")
    print("=" * 80)
    print()
    
    # Check if L1 has been run
    equations_path = os.path.join(root_dir, 'Output', 'L1_equations.txt')
    if not os.path.exists(equations_path):
        print("ERROR: L1 equations file not found!")
        print(f"Expected: {equations_path}")
        print("\nPlease run Lab 1 first:")
        print("  run_labs.bat → Select [1] Lab 1")
        return 1
    
    print(f"Loading equations from: {equations_path}")
    
    # Step 1: Parse equations from L1
    print("\n[Step 1] Parsing equations from L1...")
    try:
        data = load_from_L1(equations_path)
        print(f"  Loaded {data['n_states']} states")
        print(f"  Absorbing states: {[s + 1 for s in data['absorbing_states']]}")
    except Exception as e:
        print(f"ERROR: Failed to parse equations: {e}")
        return 1
    
    output_dir = os.path.join(root_dir, 'Output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 2: Load L2 solution for comparison
    print("\n[Step 2] Loading L2 analytical solution...")
    l2_solution = ensure_l2_solution()
    
    if l2_solution:
        print(f"  L2 solution loaded: {l2_solution['n_states']} states")
        print(f"  Time range: [{l2_solution['t'][0]:.2f}, {l2_solution['t'][-1]:.2f}]")
        print(f"  Grid points: {len(l2_solution['t'])}")
    else:
        print("  WARNING: L2 solution not available!")
    
    # Step 3: Solve using Modified Euler method
    print("\n[Step 3] Solving using Modified Euler method...")
    
    # Base step size
    base_h = 0.01
    t_span = (0, 30)
    
    try:
        solver = ModifiedEulerSolver(
            Q=data['Q'],
            initial_state=data['initial_state'],
            h=base_h,
            t_span=t_span
        )
        
        print(f"  Step size: h = {base_h}")
        print(f"  Time span: {t_span}")
        print("  Computing...", end=" ")
        
        solution = solver.solve(store_steps=500)
        
        print(f"Done ({solution['n_steps']} steps, {solution['execution_time']*1000:.2f} ms)")
        print(f"  Max probability deviation: {solution['max_prob_deviation']:.2e}")
        
    except Exception as e:
        print(f"ERROR: Failed to solve: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 4: Plot numerical solution
    print("\n[Step 4] Plotting numerical solution...")
    try:
        plot_numerical_solution(solver, l2_solution, output_dir)
    except Exception as e:
        print(f"WARNING: Plotting failed: {e}")
    
    # Step 5: Compare with L2
    print("\n[Step 5] Comparing with L2 analytical solution...")
    comparison_results = None
    
    if l2_solution is not None:
        try:
            comparator = L3L2Comparator(solver, l2_solution)
            comparison_results = comparator.compare()
            
            print(f"  Max absolute error: {comparison_results['global_metrics']['max_abs_error']:.6e}")
            print(f"  Max relative error: {comparison_results['global_metrics']['max_rel_error']:.6e}")
            print(f"  RMSE: {comparison_results['global_metrics']['rmse']:.6e}")
            
        except Exception as e:
            print(f"WARNING: Comparison failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  Skipped (L2 solution not available)")
    
    # Step 6: Generate report
    print("\n[Step 6] Generating report...")
    try:
        report = generate_report(solver, comparison_results)
        
        # Save report
        report_file = os.path.join(output_dir, 'L3_results.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"  Saved report: {report_file}")
        
        # Print preview
        print("\nReport preview (first 40 lines):")
        print("-" * 50)
        for line in report.split('\n')[:40]:
            print(line)
        if len(report.split('\n')) > 40:
            print("...")
        
    except Exception as e:
        print(f"ERROR: Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print("\n" + "=" * 80)
    print("LAB 3 COMPLETED")
    print("=" * 80)
    print("\nGenerated files in Output/:")
    print("  [PNG] L3_probabilities.png - Numerical solution plot")
    print("  [TXT] L3_results.txt - Detailed report")
    print()
    
    if comparison_results:
        print("Error Summary:")
        print(f"  Max absolute error: {comparison_results['global_metrics']['max_abs_error']:.6e}")
        print(f"  RMSE: {comparison_results['global_metrics']['rmse']:.6e}")
    
    print()
    print("Modified Euler method implementation complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

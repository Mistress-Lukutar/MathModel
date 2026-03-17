"""
Main entry point for L2 - Operator Method Solver.
Runs the complete L2 workflow: parse, solve, compare, report.

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-16
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

from L2.equation_parser import load_from_L1
from L2.operator_solver import OperatorSolver
from L2.comparison import L2L1Comparator
from L2.report_generator import generate_report


def load_config():
    """Load configuration from root config.json."""
    config_path = os.path.join(root_dir, 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_analytical_solution(solver, t_span=(0, 30), t_points=500, output_dir=None):
    """
    Plot analytical solution.
    
    Args:
        solver (OperatorSolver): Solved solver
        t_span (tuple): Time range
        t_points (int): Number of points
        output_dir (str): Output directory
    """
    if output_dir is None:
        output_dir = os.path.join(root_dir, 'Output')
    os.makedirs(output_dir, exist_ok=True)
    
    t = np.linspace(t_span[0], t_span[1], t_points)
    y = solver.evaluate(t)
    
    # Plot all states
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, solver.n_states))
    
    for i in range(solver.n_states):
        is_absorbing = i in solver.absorbing_states
        label = f"P_{i+1}(t)" if not is_absorbing else f"P_{i+1}(t) [absorbing]"
        plt.plot(t, y[i, :], label=label, linewidth=2.5, color=colors[i])
    
    plt.xlabel("Time (t)", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.title("Analytical Solution - Operator Method (Laplace Transform)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='right', fontsize=10)
    plt.xlim(left=0)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'L2_analytical_solution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved analytical solution plot: {output_file}")
    
    return t, y


def main():
    """Main L2 workflow."""
    print("=" * 70)
    print("LAB 2: OPERATOR METHOD FOR KOLMOGOROV EQUATIONS")
    print("=" * 70)
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
    
    # Step 2: Solve using operator method
    print("\n[Step 2] Solving using operator method (Laplace transform)...")
    try:
        solver = OperatorSolver(
            Q=data['Q'],
            initial_state=data['initial_state'],
            absorbing_states=data['absorbing_states']
        )
        solver.solve()
    except Exception as e:
        print(f"ERROR: Failed to solve: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 3: Plot analytical solution
    print("\n[Step 3] Plotting analytical solution...")
    try:
        t, y = plot_analytical_solution(solver)
    except Exception as e:
        print(f"WARNING: Plotting failed: {e}")
        t, y = None, None
    
    # Step 4: Compare with L1 (if available)
    print("\n[Step 4] Comparing with L1 numerical solution...")
    comparison_results = None
    try:
        comparator = L2L1Comparator(solver)
        comparison_results = comparator.compare()
        if comparison_results:
            comparator.plot_comparison()
        else:
            print("  Comparison skipped (L1 solution not available)")
    except Exception as e:
        print(f"WARNING: Comparison failed: {e}")
    
    # Step 5: Generate report
    print("\n[Step 5] Generating report...")
    try:
        report = generate_report(solver, comparison_results)
        print("\nReport preview (first 50 lines):")
        print("-" * 50)
        for line in report.split('\n')[:50]:
            print(line)
        if len(report.split('\n')) > 50:
            print("...")
    except Exception as e:
        print(f"ERROR: Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print("\n" + "=" * 70)
    print("LAB 2 COMPLETED")
    print("=" * 70)
    print("\nGenerated files in Output/:")
    print("  [PNG] L2_analytical_solution.png - Analytical solution plot")
    if comparison_results:
        print("  [PNG] L2_comparison.png - Comparison with L1")
    print("  [TXT] L2_results.txt - Detailed report with formulas")
    print()
    print("Analytical formulas available in L2_results.txt")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

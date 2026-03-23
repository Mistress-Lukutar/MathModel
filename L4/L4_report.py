"""
Main entry point for L4 - Accuracy Analysis.

Runs the complete L4 workflow:
1. Load analytical solution from L2
2. Load/Compute numerical solutions with different step sizes
3. Analyze accuracy and convergence
4. Analyze timing
5. Generate comprehensive report and plots

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
from L4.accuracy_analyzer import AccuracyAnalyzer
from L4.convergence_analysis import ConvergenceAnalyzer
from L4.step_comparison import StepComparator
from L4.timing_analyzer import TimingAnalyzer
from L4.report_generator import generate_report, save_report


def load_config():
    """Load configuration from root config.json."""
    config_path = os.path.join(root_dir, 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_prerequisites():
    """
    Check if required files from previous labs exist.
    
    Returns:
        bool: True if all prerequisites met
    """
    required_files = [
        ('L1 equations', os.path.join(root_dir, 'Output', 'L1_equations.txt')),
        ('L2 analytical solution', os.path.join(root_dir, 'Output', 'L2_solution.npy')),
    ]
    
    all_ok = True
    for name, path in required_files:
        if os.path.exists(path):
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: NOT FOUND - {path}")
            all_ok = False
    
    return all_ok


def load_l2_solution():
    """Load analytical solution from L2."""
    l2_path = os.path.join(root_dir, 'Output', 'L2_solution.npy')
    data = np.load(l2_path, allow_pickle=True).item()
    print(f"  Loaded L2 solution: {data['n_states']} states, "
          f"{len(data['t'])} time points")
    return data


def analyze_base_accuracy(Q, initial_state, l2_data, base_h=0.01):
    """
    Analyze accuracy of numerical solution with base step size.
    
    Args:
        Q: Intensity matrix
        initial_state: Initial state vector
        l2_data: L2 analytical solution data
        base_h: Base step size
        
    Returns:
        AccuracyAnalyzer instance with results
    """
    print("\n[Step 3] Analyzing accuracy with base step size...")
    
    # Solve with base step
    solver = ModifiedEulerSolver(Q, initial_state, h=base_h, t_span=(0, 30))
    solution = solver.solve(store_steps=500)
    
    print(f"  Step size: h = {base_h}")
    print(f"  Steps computed: {solution['n_steps']}")
    print(f"  Execution time: {solution['execution_time']*1000:.2f} ms")
    
    # Analyze accuracy
    analyzer = AccuracyAnalyzer(
        t_num=solver.t_values,
        y_num=solver.P_values,
        t_anal=l2_data['t'],
        y_anal=l2_data['y']
    )
    
    metrics = analyzer.get_all_metrics()
    print(f"  Max absolute error: {metrics['global']['max_abs_error']:.6e}")
    print(f"  Max relative error: {metrics['global']['max_rel_error']:.6e}")
    
    return analyzer


def run_convergence_analysis(Q, initial_state, l2_data, step_sizes):
    """
    Run convergence analysis with different step sizes.
    
    Args:
        Q: Intensity matrix
        initial_state: Initial state vector
        l2_data: L2 analytical solution data
        step_sizes: List of step sizes to test
        
    Returns:
        Tuple of (ConvergenceAnalyzer, order)
    """
    print("\n[Step 4] Running convergence analysis...")
    
    conv_analyzer = ConvergenceAnalyzer(
        Q=Q,
        initial_state=initial_state,
        t_span=(0, 30),
        reference_solution=l2_data
    )
    
    results = conv_analyzer.run_convergence_study(step_sizes, store_steps=500)
    order = conv_analyzer.estimate_convergence_order()
    
    if order:
        print(f"  Estimated convergence order: O(h^{order:.2f})")
    
    return conv_analyzer, order


def analyze_timing(conv_results):
    """
    Analyze timing characteristics.
    
    Args:
        conv_results: Convergence analysis results
        
    Returns:
        Tuple of (timing_analysis, efficiency_metrics, optimal_steps)
    """
    print("\n[Step 5] Analyzing timing...")
    
    timing_analyzer = TimingAnalyzer()
    
    timing_analysis = timing_analyzer.analyze_timing_vs_step(conv_results)
    efficiency_metrics = timing_analyzer.compute_efficiency_metrics(conv_results)
    
    optimal_steps = {}
    for criterion in ['accuracy', 'speed', 'balanced']:
        optimal_steps[criterion] = timing_analyzer.find_optimal_step(
            conv_results, criterion
        )
    
    print(f"  Complexity: time ~ h^{timing_analysis['complexity_slope']:.2f}")
    print(f"  Optimal step (balanced): h = {optimal_steps['balanced']['h']:.4f}" 
          if optimal_steps['balanced'] else "  N/A")
    
    return timing_analysis, efficiency_metrics, optimal_steps


def generate_plots(conv_analyzer, step_comparator, timing_analyzer,
                   accuracy_analyzer, output_dir, group_results=None):
    """
    Generate all 5 plots for Lab 4.
    
    Args:
        conv_analyzer: ConvergenceAnalyzer instance
        step_comparator: StepComparator instance
        timing_analyzer: TimingAnalyzer instance
        accuracy_analyzer: AccuracyAnalyzer instance
        output_dir: Output directory
        group_results: Optional two-group analysis results
    """
    print("\n[Step 6] Generating plots (5 figures)...")
    
    # Plot 1: Convergence for coarse steps (Group 1)
    if group_results and 'coarse' in group_results:
        print("  Plot 1/5: Convergence analysis (coarse steps)...")
        coarse_comparator = StepComparator(
            group_results['coarse']['results'], 
            reference_solution=conv_analyzer.reference
        )
        coarse_comparator.plot_error_vs_step(output_dir, suffix="coarse")
    
    # Plot 2: Convergence for fine steps (Group 2)
    if group_results and 'fine' in group_results:
        print("  Plot 2/5: Convergence analysis (fine steps)...")
        fine_comparator = StepComparator(
            group_results['fine']['results'],
            reference_solution=conv_analyzer.reference
        )
        fine_comparator.plot_error_vs_step(output_dir, suffix="fine")
    
    # Plot 3: Accuracy analysis (single plot for all)
    print("  Plot 3/5: Accuracy analysis...")
    _plot_accuracy_analysis(accuracy_analyzer, output_dir)
    
    # Plot 4: Timing for coarse steps
    if group_results and 'coarse' in group_results:
        print("  Plot 4/5: Timing analysis (coarse steps)...")
        timing_analyzer.plot_timing_analysis(
            group_results['coarse']['results'], 
            output_dir, 
            suffix="coarse"
        )
    
    # Plot 5: Timing for fine steps
    if group_results and 'fine' in group_results:
        print("  Plot 5/5: Timing analysis (fine steps)...")
        timing_analyzer.plot_timing_analysis(
            group_results['fine']['results'],
            output_dir,
            suffix="fine"
        )


def _plot_accuracy_analysis(analyzer, output_dir):
    """Plot accuracy analysis results."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top: Error over time (all states)
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, analyzer.n_states))
    
    for i in range(analyzer.n_states):
        ax.semilogy(analyzer.t_num, analyzer.abs_error[i, :] + 1e-16,
                   color=colors[i], linewidth=1.5, label=f'Error P_{i+1}')
    
    ax.set_xlabel('Time (t)', fontsize=11)
    ax.set_ylabel('Absolute Error', fontsize=11)
    ax.set_title('Absolute Error Evolution', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(left=0)
    
    # Bottom: Relative error
    ax = axes[1]
    for i in range(analyzer.n_states):
        valid_mask = analyzer.rel_error[i, :] > 0
        if np.any(valid_mask):
            ax.semilogy(analyzer.t_num[valid_mask], 
                       analyzer.rel_error[i, valid_mask] + 1e-16,
                       color=colors[i], linewidth=1.5, label=f'Rel Error P_{i+1}')
    
    ax.set_xlabel('Time (t)', fontsize=11)
    ax.set_ylabel('Relative Error', fontsize=11)
    ax.set_title('Relative Error Evolution', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'L4_accuracy_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved accuracy analysis plot: {output_file}")


def _plot_timing_analysis(timing_analysis, output_dir):
    """Plot timing analysis results."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    steps = timing_analysis['steps']
    times = timing_analysis['times']
    
    # Left: Time vs step
    ax = axes[0]
    ax.plot(steps, times * 1000, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Step Size (h)', fontsize=11)
    ax.set_ylabel('Execution Time (ms)', fontsize=11)
    ax.set_title('Execution Time vs Step Size', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Right: Time vs 1/h (should be linear)
    ax = axes[1]
    inv_h = 1.0 / steps
    ax.plot(inv_h, times * 1000, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('1/h', fontsize=11)
    ax.set_ylabel('Execution Time (ms)', fontsize=11)
    ax.set_title('Execution Time vs 1/h (Linear Check)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Fit line
    if len(steps) >= 2:
        A = np.vstack([inv_h, np.ones(len(inv_h))]).T
        slope, intercept = np.linalg.lstsq(A, times * 1000, rcond=None)[0]
        inv_h_fit = np.linspace(min(inv_h), max(inv_h), 100)
        time_fit = slope * inv_h_fit + intercept
        ax.plot(inv_h_fit, time_fit, 'g--', linewidth=1.5, 
               alpha=0.7, label=f'Fit: t = {slope:.3f}/h + {intercept:.3f}')
        ax.legend(fontsize=9)
    
    plt.suptitle('Timing Analysis - Modified Euler Method', fontsize=13)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'L4_timing.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved timing plot: {output_file}")


def main():
    """Main L4 workflow."""
    print("=" * 80)
    print("LAB 4: ACCURACY ANALYSIS OF NUMERICAL SOLUTIONS")
    print("Variant 8: Modified Euler Method")
    print("=" * 80)
    print()
    
    # Check prerequisites
    print("[Step 1] Checking prerequisites...")
    if not check_prerequisites():
        print("\nERROR: Missing required files from previous labs!")
        print("Please run Lab 1 and Lab 2 first:")
        print("  run_labs.bat → Select [1] Lab 1")
        print("  run_labs.bat → Select [2] Lab 2")
        return 1
    
    # Load configuration
    config = load_config()
    l4_config = config.get('L4', {})
    
    # Get step sizes from config
    step_sizes = l4_config.get('analysis', {}).get(
        'convergence_steps', [0.04, 0.02, 0.01, 0.005]
    )
    
    # Load L1 data
    print("\n[Step 2] Loading system data from L1...")
    equations_path = os.path.join(root_dir, 'Output', 'L1_equations.txt')
    l1_data = load_from_L1(equations_path)
    print(f"  Loaded {l1_data['n_states']} states")
    
    # Load L2 solution
    print("\n  Loading analytical solution from L2...")
    l2_data = load_l2_solution()
    
    # Setup output
    output_dir = os.path.join(root_dir, 'Output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 3: Base accuracy analysis
    try:
        accuracy_analyzer = analyze_base_accuracy(
            l1_data['Q'], l1_data['initial_state'], l2_data
        )
        accuracy_metrics = accuracy_analyzer.get_all_metrics()
    except Exception as e:
        print(f"ERROR: Accuracy analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 4: Convergence analysis with TWO groups
    print("\n[Step 4] Running convergence analysis with two step groups...")
    print("  Group 1 (coarse): truncation error dominates")
    print("  Group 2 (fine): round-off error affects convergence")
    
    # Get both step groups from config
    coarse_steps = step_sizes  # [0.5, 0.4, 0.3, 0.2, 0.1]
    fine_steps = l4_config.get('analysis', {}).get(
        'fine_steps', [0.16, 0.08, 0.04, 0.02, 0.01]
    )
    
    try:
        conv_analyzer = ConvergenceAnalyzer(
            Q=l1_data['Q'],
            initial_state=l1_data['initial_state'],
            t_span=(0, 30),
            reference_solution=l2_data
        )
        
        # Run two-group analysis
        group_results = conv_analyzer.analyze_step_groups(
            coarse_steps=coarse_steps,
            fine_steps=fine_steps,
            store_steps=500
        )
        
        order = group_results['coarse']['order']  # Use coarse for main report
        
    except Exception as e:
        print(f"ERROR: Convergence analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 5: Timing analysis
    try:
        timing_analysis, efficiency_metrics, optimal_steps = analyze_timing(
            conv_analyzer.results
        )
    except Exception as e:
        print(f"ERROR: Timing analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 6: Generate plots (5 figures)
    try:
        step_comparator = StepComparator(
            conv_analyzer.results, reference_solution=l2_data
        )
        timing_analyzer = TimingAnalyzer()
        generate_plots(conv_analyzer, step_comparator, timing_analyzer,
                      accuracy_analyzer, output_dir, group_results)
    except Exception as e:
        print(f"WARNING: Plot generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 7: Generate report
    print("\n[Step 7] Generating report...")
    try:
        report = generate_report(
            accuracy_metrics=accuracy_metrics,
            convergence_results=conv_analyzer.results,
            convergence_order=order,
            timing_analysis=timing_analysis,
            efficiency_metrics=efficiency_metrics,
            optimal_steps=optimal_steps,
            variant=l4_config.get('variant', 8),
            group_results=group_results
        )
        
        # Save report
        report_file = os.path.join(output_dir, 'L4_results.txt')
        save_report(report, report_file)
        
        # Print preview
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
    print("\n" + "=" * 80)
    print("LAB 4 COMPLETED")
    print("=" * 80)
    print("\nGenerated files in Output/:")
    print("  [PNG] L4_convergence_coarse.png - Convergence plot (coarse steps)")
    print("  [PNG] L4_convergence_fine.png - Convergence plot (fine steps)")
    print("  [PNG] L4_accuracy_analysis.png - Error evolution analysis")
    print("  [PNG] L4_timing_coarse.png - Timing analysis (coarse steps)")
    print("  [PNG] L4_timing_fine.png - Timing analysis (fine steps)")
    print("  [TXT] L4_results.txt - Comprehensive report")
    print()
    
    if order:
        print("Convergence Analysis:")
        print(f"  Estimated order: O(h^{order:.2f})")
        print(f"  Theoretical order: O(h^2.00)")
    
    print("\nKey Findings:")
    print(f"  Max absolute error (h=0.01): {accuracy_metrics['global']['max_abs_error']:.6e}")
    if 'balanced' in optimal_steps and optimal_steps['balanced']:
        print(f"  Optimal step size: h = {optimal_steps['balanced']['h']:.4f}")
    
    print("\nAccuracy analysis complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

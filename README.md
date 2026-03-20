# MathModel

Mathematical modeling educational project consisting of multiple lab works for a course on modeling technical objects and control systems.

## Labs

| Lab | Topic | Status |
|-----|-------|--------|
| Lab 1 | Continuous-time Markov Chains | ✅ Implemented |
| Lab 2 | Operator Method (Laplace Transform) | ✅ Implemented |
| Lab 3 | Numerical Solution (Modified Euler) | ✅ Implemented |
| Lab 4 | [Placeholder] | 🚧 Not implemented |

## Technology Stack

- **Language:** Python 3.8+
- **Numerical Computing:** NumPy
- **ODE Solver:** SciPy (`scipy.integrate.solve_ivp`)
- **Linear Algebra:** SciPy (`scipy.linalg`)
- **Visualization:** Matplotlib
- **Graph Visualization:** NetworkX, Graphviz (neato)

## Project Structure

```
MathModel/
├── .venv/                    # Virtual environment
├── L1/                       # Lab 1: Markov Chains
│   ├── markov_solver.py      # Main solver: builds matrix, solves ODEs
│   ├── markov_graph.py       # Graph visualization
│   └── stationary_check.py   # Absorption probability analysis
├── L2/                       # Lab 2: Operator Method
│   ├── __init__.py           # Package initialization
│   ├── equation_parser.py    # Parse L1 equations export
│   ├── operator_solver.py    # Laplace transform solver
│   ├── comparison.py         # L2 vs L1 comparison
│   ├── report_generator.py   # Text report generation
│   └── L2_report.py          # Main entry point
├── L3/                       # Lab 3: Numerical Solution
│   ├── __init__.py           # Package initialization
│   ├── modified_euler.py     # Modified Euler method implementation
│   ├── comparison.py         # L3 vs L2 comparison
│   ├── report_generator.py   # Text report generation
│   └── L3_report.py          # Main entry point
├── L4/                       # Lab 4: [Placeholder]
├── Output/                   # Generated output files (PNG, etc.)
├── config.json               # Central configuration for all labs
├── requirements.txt          # Python dependencies
├── run_labs.bat             # Main launcher with menu
└── AGENTS.md                # Agent documentation
```

## Quick Start (Windows)

```batch
run_labs.bat
```

The batch script will:
1. Check for Python 3.8+
2. Create virtual environment `.venv/` at project root if not exists
3. Install dependencies from `requirements.txt`
4. Show menu with available labs
5. Execute selected lab workflow
6. Save outputs to `Output/` directory

## Lab 1: Markov Chains

### Features

- Builds intensity matrix Q from configuration
- Solves Kolmogorov forward equations using RK45 method
- Generates state transition graph (PNG)
- Plots probability evolution over time
- Calculates absorption probabilities using fundamental matrix method
- Validates probability conservation

### Configuration

Edit `config.json` to customize:

```json
{
  "L1": {
    "n_states": 7,
    "initial_state": 0,
    "transitions": [
      {"from": 1, "to": 4, "rate": 0.1},
      ...
    ]
  }
}
```

### Outputs

| File | Description |
|------|-------------|
| `Output/L1_markov_graph.png` | State transition diagram |
| `Output/L1_probabilities.png` | Probability evolution plot |
| `Output/L1_results.txt` | Full calculation report |

## Lab 2: Operator Method

### Features

- Parses differential equations exported from Lab 1
- Solves algebraic system in p-domain using Laplace transform
- Performs partial fraction decomposition for inverse transforms
- Compares analytical (L2) vs numerical (L1) solutions
- Generates detailed comparison reports with error metrics
- Exports solution data for Lab 3 comparison

### Outputs

| File | Description |
|------|-------------|
| `Output/L2_analytical_solution.png` | Analytical solution plot |
| `Output/L2_comparison.png` | L1 vs L2 solution comparison plot |
| `Output/L2_results.txt` | Detailed report with formulas |
| `Output/L2_solution.npy` | Solution data for Lab 3 |

## Lab 3: Numerical Solution (Modified Euler Method)

### Features

- Solves Kolmogorov equations using Modified Euler method (predictor-corrector)
- Custom numerical implementation without scipy.integrate
- Second-order accuracy O(h²)
- Compares numerical (L3) vs analytical (L2) solutions
- Computes absolute and relative errors
- Finds interval of maximum deviation

### Outputs

| File | Description |
|------|-------------|
| `Output/L3_probabilities.png` | Numerical solution vs L2 analytical |
| `Output/L3_results.txt` | Detailed report with error analysis |

## Authors

- **Mistress-Lukutar** - Initial work

## Version

1.1 (2026-03-16)

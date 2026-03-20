# AGENTS.md — MathModel Project

This document provides essential information for AI coding agents working on this project.

---

## Agent Workspace (`.agents/`)

Directory `.agents/` is designated for agent-related files and is excluded from git tracking:

- **Purpose:** Store plans, temporary files, notes, and draft documents
- **Status:** Listed in `.gitignore` — never committed to repository
- **Usage:** 
  - Place task plans, implementation drafts, research notes here
  - Store extracted/analyzed data temporarily
  - Keep work-in-progress documents

**Example contents:**
```
.agents/
├── L2_implementation_plan.md
├── research_notes.txt
├── temp_analysis.json
└── ...
```

---

## Project Overview

---

## Project Overview

This is a **mathematical modeling educational project** consisting of multiple lab works for a course on modeling technical objects and control systems.

**Current Labs:**
| Lab | Topic | Status |
|-----|-------|--------|
| Lab 1 | Continuous-time Markov Chains | ✅ Implemented |
| Lab 2 | Operator Method (Laplace Transform) | ✅ Implemented |
| Lab 3 | Numerical Solution (Modified Euler) | ✅ Implemented |
| Lab 4 | [Placeholder] | 🚧 Not implemented |

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.8+ |
| Numerical Computing | NumPy |
| ODE Solver | SciPy (`scipy.integrate.solve_ivp`) |
| Linear Algebra | SciPy (`scipy.linalg`) |
| Visualization | Matplotlib |
| Graph Visualization | NetworkX |

### Dependencies

All dependencies are listed in `requirements.txt`:
```
numpy
scipy
matplotlib
networkx
```

---

## Project Structure

```
MathModel/
├── .venv/                    # Virtual environment (created by run_labs.bat)
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
├── L3/                       # Lab 3: Numerical Solution (Modified Euler)
│   ├── __init__.py           # Package initialization
│   ├── modified_euler.py     # Modified Euler method implementation
│   ├── step_analysis.py      # Convergence analysis with different steps
│   ├── comparison.py         # L3 vs L2 comparison
│   ├── report_generator.py   # Text report generation
│   └── L3_report.py          # Main entry point
├── L4/                       # Lab 4: [Placeholder]
├── Output/                   # Generated output files (PNG, etc.)
│   ├── L1_markov_graph.png   # Lab 1 graph output
│   └── L1_probabilities.png  # Lab 1 probabilities plot
├── config.json               # Central configuration for all labs
├── requirements.txt          # Python dependencies
├── run_labs.bat             # Main launcher with menu
└── AGENTS.md                # This file
```

### Configuration File (`config.json`)

Central configuration file in the root directory containing settings for all labs:

```json
{
  "L1": {
    "n_states": 7,
    "initial_state": 0,
    "transitions": [...]
  },
  "L2": { "enabled": false },
  "L3": { "enabled": false },
  "L4": { "enabled": false }
}
```

- Each lab has its own section (L1, L2, L3, L4)
- Placeholder labs have `enabled: false` flag
- Lab 1 configuration:
  - `n_states`: Number of states in the Markov chain
  - `initial_state`: Starting state index (0-based)
  - `transitions`: List of transitions with `from`, `to`, and `rate` fields
  - Absorbing states are detected automatically

### Output Directory (`Output/`)

All generated files are saved to the `Output/` directory with lab-specific prefixes:
- `L1_markov_graph.png` - State transition diagram
- `L1_probabilities.png` - Probability evolution plot
- Future labs will use `L2_*`, `L3_*`, `L4_*` prefixes

### Lab 1: Markov Chains (L1/)

#### `markov_solver.py`
- **Class:** `MarkovChainSolver`
- **Purpose:** Core computation engine
- **Key Methods:**
  - `_load_config()` — Loads L1 config from root config.json
  - `_build_intensity_matrix()` — Constructs Q matrix from config
  - `_detect_absorbing_states()` — Detects absorbing states dynamically
  - `validate_matrix()` — Validates row sums = 0, off-diagonal ≥ 0, diagonal ≤ 0
  - `kolmogorov_equations()` — Returns dP/dt = P · Q
  - `solve()` — Numerical integration using RK45 method
  - `plot_probabilities()` — Generates probability evolution plot (saves to Output/L1_probabilities.png)
  - `print_differential_equations()` — Outputs LaTeX-ready equations
  - `export_for_L2()` — Exports equations for L2 operator method
  - `save_results()` — Saves results and L1_solution.npy for L2 comparison

#### `markov_graph.py`
- **Function:** `load_config()` — Loads L1 config from root config.json
- **Function:** `create_markov_graph(config)` — Creates directed graph from config
- **Function:** `plot_markov_graph(config)` — Generates state diagram (saves to Output/L1_markov_graph.png)
- **Function:** `detect_absorbing_states(config)` — Detects absorbing states from transitions

### Lab 2: Operator Method (L2/)

#### `operator_solver.py`
- **Class:** `OperatorSolver`
- **Purpose:** Analytical solution using Laplace transform
- **Key Methods:**
  - `_build_operator_system()` — Builds (pI - Q^T) system
  - `_solve_algebraic()` — Solves algebraic system in p-domain
  - `_decompose_fractions()` — Partial fraction decomposition
  - `_inverse_transform()` — Inverse Laplace transform
  - `evaluate()` — Evaluate solution at time points
  - `get_steady_state()` — Calculate steady-state probabilities

#### `equation_parser.py`
- **Class:** `EquationParser`
- **Purpose:** Parse L1 exported equations
- **Key Methods:**
  - `parse()` — Parse L1_equations.txt
  - `_parse_equations()` — Extract differential equations
  - `_build_matrix()` — Reconstruct Q matrix

#### `comparison.py`
- **Class:** `L2L1Comparator`
- **Purpose:** Compare analytical (L2) vs numerical (L1)
- **Key Methods:**
  - `load_L1_solution()` — Load L1_solution.npy
  - `compare()` — Calculate error metrics
  - `plot_comparison()` — Generate comparison plots

#### `L2_report.py`
- **Purpose:** Main entry point for Lab 2
- **Workflow:** Parse → Solve → Compare → Report → Export for L3

### Lab 3: Numerical Solution (L3/)

**Variant 8: Modified Euler Method (Heun's Method)**

Purpose: Solve the same Kolmogorov equations from L1 using a custom numerical method implementation (without scipy.integrate), and compare with the analytical L2 solution.

#### `modified_euler.py`
- **Class:** `ModifiedEulerSolver`
- **Purpose:** Numerical solution using Modified Euler method (predictor-corrector)
- **Key Methods:**
  - `f(t, P)` — Right-hand side: dP/dt = P · Q
  - `step(t, P)` — Single Modified Euler step (k1, k2, update)
  - `solve()` — Full integration with probability conservation
  - `evaluate(t_query)` — Interpolate solution at query points
- **Order of Accuracy:** O(h²) - second order

#### `step_analysis.py`
- **Class:** `StepConvergenceAnalyzer`
- **Purpose:** Analyze convergence with different step sizes
- **Key Methods:**
  - `run_with_steps(h_values)` — Run solver with multiple step sizes
  - `compute_errors(reference)` — Compute errors vs analytical solution
  - `estimate_convergence_order()` — Estimate empirical convergence order
  - `plot_convergence()` — Visualize convergence
- **Tested Steps:** h, h/2, h/3, h/4 (typically: 0.04, 0.02, 0.01, 0.005)

#### `comparison.py`
- **Class:** `L3L2Comparator`
- **Purpose:** Compare L3 numerical vs L2 analytical solutions
- **Key Methods:**
  - `compare()` — Compute absolute and relative errors
  - `find_max_deviation_interval()` — Find interval of largest error
  - `plot_comparison()` — Generate comparison plots
- **Input:** `Output/L2_solution.npy` (exported from L2)

#### `report_generator.py`
- **Class:** `L3ReportGenerator`
- **Purpose:** Generate comprehensive report
- **Sections:** Theory, Input Data, Results, Error Analysis, Convergence, Conclusions

#### `L3_report.py`
- **Purpose:** Main entry point for Lab 3
- **Workflow:** Load L1 → Solve L3 → Compare with L2 → Convergence Analysis → Report

---

#### `stationary_check.py`
- **Function:** `load_config()` — Loads L1 config from root config.json
- **Function:** `analyze_absorbing_states(config)` — Computes absorption probabilities
- **Function:** `build_matrices_from_config(config)` — Builds Q_transient and R matrices

---

## Build and Run Instructions

### Quick Start (Windows)

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

### Manual Execution

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate.bat  # Windows
# or: source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt

# Run Lab 1
cd L1
python markov_graph.py       # Step 1: Generate graph -> Output/L1_markov_graph.png
python markov_solver.py      # Step 2: Solve ODEs -> Output/L1_probabilities.png
python stationary_check.py   # Step 3: Analyze absorption probabilities
```

### Expected Outputs (Lab 1)

| File | Description |
|------|-------------|
| `Output/L1_markov_graph.png` | State transition diagram |
| `Output/L1_probabilities.png` | Probability evolution over time |
| `Output/L1_results.txt` | Full calculation report |
| `Output/L1_equations.txt` | Exported equations for L2 |

### Expected Outputs (Lab 2)

| File | Description |
|------|-------------|
| `Output/L2_analytical_solution.png` | Analytical solution plot |
| `Output/L2_comparison.png` | Comparison with L1 numerical |
| `Output/L2_results.txt` | Detailed report with formulas |
| `Output/L2_solution.npy` | Solution data for L3 comparison |
| `Output/L2_formulas.txt` | Analytical formulas reference |

### Expected Outputs (Lab 3)

| File | Description |
|------|-------------|
| `Output/L3_probabilities.png` | Numerical solution plot |
| `Output/L3_comparison.png` | Comparison with L2 analytical |
| `Output/L3_convergence.png` | Convergence analysis plot |
| `Output/L3_results.txt` | Detailed report with error analysis |

---

## Configuration Format

### Root Configuration (`config.json`)

```json
{
  "L1": {
    "n_states": 7,
    "initial_state": 0,
    "transitions": [
      {"from": 1, "to": 4, "rate": 0.1},
      ...
    ]
  },
  "L2": {
    "enabled": false,
    "...": "..."
  }
}
```

- `from` and `to` are 1-based state indices
- `rate` is the transition intensity λ
- Absorbing states are detected automatically (no outgoing transitions)

---

## Code Style Guidelines

### Documentation Style
- **Docstrings:** Google-style docstrings with Args/Returns sections
- **Comments:** Primarily English
- **Type hints:** Not used (follows NumPy/SciPy conventions)

### Naming Conventions
- **Classes:** `PascalCase` (e.g., `MarkovChainSolver`)
- **Methods/Functions:** `snake_case` (e.g., `validate_matrix`)
- **Output files:** `L{lab_number}_{description}.png`

### Mathematical Conventions
- States are **1-indexed in comments and output**, **0-indexed in code**
- Matrix Q follows standard infinitesimal generator convention:
  - Off-diagonal Q[i,j] = λᵢⱼ (transition rate from i to j)
  - Diagonal Q[i,i] = -Σⱼ Q[i,j] (negative sum of outgoing rates)

---

## Testing Strategy

**No formal test suite exists.** Validation is built into the main modules:

### Built-in Validation Checks

1. **Matrix Validation** (`markov_solver.py`):
   - Row sums must be approximately zero
   - Off-diagonal elements must be non-negative
   - Diagonal elements must be non-positive

2. **Solution Verification** (`markov_solver.py`):
   - Probability conservation check: ΣPᵢ(t) ≈ 1.0 at all times

3. **Absorption Probability Check** (`stationary_check.py`):
   - Probabilities to absorbing states should sum to 1.0

---

## Development Notes

### Adding a New Lab

To add a new lab (e.g., Lab 2):

1. **Create directory:** `L3/`
2. **Add configuration section** to `config.json`:
   ```json
   "L3": {
     "enabled": true,
     "...": "your config here"
   }
   ```
3. **Add Python scripts:** Implement the lab logic in `L3/`
4. **Update output paths:** Scripts should save to `Output/L3_*`
5. **Update `run_labs.bat`:** Add menu entry and execution logic
6. **Update `AGENTS.md`:** Document the new lab

### Modifying Lab Configuration

To adapt Lab 1 for different transition rates or state count:

1. **Edit `config.json`:**
   - Update `L1.n_states`
   - Modify `L1.transitions` list
   - Adjust `L1.initial_state` if needed

2. **Scripts automatically:**
   - Detect absorbing states from transitions
   - Build matrices from configuration
   - Generate outputs to `Output/L1_*.png`

---

## Language Notes

- **Source code comments:** English
- **User-facing output:** English
- **Documentation:** English

# MathModel

Mathematical modeling educational project consisting of multiple lab works for a course on modeling technical objects and control systems.

## Labs

| Lab | Topic | Status |
|-----|-------|--------|
| Lab 1 | Continuous-time Markov Chains | ✅ Implemented |
| Lab 2 | [Placeholder] | 🚧 Not implemented |
| Lab 3 | [Placeholder] | 🚧 Not implemented |
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
├── L2/                       # Lab 2: [Placeholder]
├── L3/                       # Lab 3: [Placeholder]
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

## Authors

- **Mistress-Lukutar** - Initial work

## Version

1.0 (2026-03-16)

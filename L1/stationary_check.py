"""
Stationary distribution analysis for absorbing Markov chains.
Reads configuration from root config.json and verifies that probability 
concentrates in absorbing states.

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-16
"""

import json
import os
import numpy as np
from scipy.linalg import null_space


def load_config(config_path=None):
    """
    Load L1 configuration from root config.json.
    
    Args:
        config_path (str): Path to config file. If None, uses root config.json
        
    Returns:
        dict: L1 configuration section
    """
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        config_path = os.path.join(root_dir, 'config.json')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = json.load(f)
    
    return full_config['L1']


def detect_absorbing_states(config):
    """
    Detect absorbing states from transitions.
    Absorbing states have no outgoing transitions.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
            list: List of absorbing state indices (0-based)
    """
    n = config['n_states']
    has_outgoing = set()
    
    for tr in config['transitions']:
        has_outgoing.add(tr['from'])
    
    # Convert to 0-based indices
    absorbing = [i - 1 for i in range(1, n + 1) if i not in has_outgoing]
    return absorbing


def build_matrices_from_config(config):
    """
    Build Q_transient and R matrices from configuration.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (Q_transient, R, transient_indices, absorbing_indices)
    """
    n = config['n_states']
    
    # Detect absorbing states dynamically
    absorbing_indices = detect_absorbing_states(config)
    transient_indices = [i for i in range(n) if i not in absorbing_indices]
    
    # Build full intensity matrix Q
    Q = np.zeros((n, n))
    for tr in config['transitions']:
        i = tr['from'] - 1
        j = tr['to'] - 1
        Q[i, j] = tr['rate']
    
    for i in range(n):
        Q[i, i] = -np.sum(Q[i, :])
    
    # Extract Q_transient (transient-to-transient)
    n_transient = len(transient_indices)
    Q_transient = np.zeros((n_transient, n_transient))
    for i, ti in enumerate(transient_indices):
        for j, tj in enumerate(transient_indices):
            Q_transient[i, j] = Q[ti, tj]
    
    # Extract R (transient-to-absorbing)
    n_absorbing = len(absorbing_indices)
    R = np.zeros((n_transient, n_absorbing))
    for i, ti in enumerate(transient_indices):
        for j, aj in enumerate(absorbing_indices):
            R[i, j] = Q[ti, aj]
    
    return Q_transient, R, transient_indices, absorbing_indices


def analyze_absorbing_states(config=None, output_file=None):
    """
    Analyze long-term behavior of Markov chain using fundamental matrix method.
    
    Args:
        config (dict): Configuration dictionary. If None, loads from root config.json
        output_file (str): Path to save results. If None, appends to Output/L1_results.txt
        
    Returns:
        dict: Analysis results
    """
    if config is None:
        config = load_config()
    
    Q_transient, R, transient_indices, absorbing_indices = build_matrices_from_config(config)
    
    # Fundamental matrix N = (-Q)^(-1)
    N = np.linalg.inv(-Q_transient)
    
    # Absorption probabilities B = N · R
    B = N @ R
    
    # Get initial state (convert to 0-based)
    initial_state_idx = config['initial_state']
    
    # Calculate absorption probabilities from initial state
    if initial_state_idx in absorbing_indices:
        absorption_probs = {}
        for i, abs_idx in enumerate(absorbing_indices):
            if abs_idx == initial_state_idx:
                absorption_probs[abs_idx + 1] = 1.0
            else:
                absorption_probs[abs_idx + 1] = 0.0
    else:
        initial_transient_row = transient_indices.index(initial_state_idx)
        absorption_probs = {}
        for i, abs_idx in enumerate(absorbing_indices):
            absorption_probs[abs_idx + 1] = B[initial_transient_row, i]
    
    # Print to console
    print(f"Absorption Probabilities from initial state {initial_state_idx + 1}:")
    for state, prob in absorption_probs.items():
        print(f"Probability of eventual absorption in state {state}: {prob:.6f}")
    
    total_prob = sum(absorption_probs.values())
    print(f"Sum check: {total_prob:.6f} (should be 1.0)")
    
    # Save to file
    if output_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        output_dir = os.path.join(root_dir, 'Output')
        output_file = os.path.join(output_dir, 'L1_results.txt')
    
    # Append to existing file
    lines = []
    lines.append("\n")
    lines.append("7. DETAILED ABSORPTION ANALYSIS (fundamental matrix method)")
    lines.append("-" * 80)
    lines.append("")
    lines.append("7.1. Matrix Q (transient → transient)")
    lines.append("Q_transient =")
    for i in range(Q_transient.shape[0]):
        row_str = "  [" + " ".join([f"{Q_transient[i,j]:7.4f}" for j in range(Q_transient.shape[1])]) + "]"
        lines.append(row_str)
    lines.append("")
    
    lines.append("7.2. Matrix R (transient → absorbing)")
    lines.append("R =")
    for i in range(R.shape[0]):
        row_str = "  [" + " ".join([f"{R[i,j]:7.4f}" for j in range(R.shape[1])]) + "]"
        lines.append(row_str)
    lines.append("")
    
    lines.append("7.3. Fundamental matrix N = (-Q)^(-1)")
    lines.append("N =")
    for i in range(N.shape[0]):
        row_str = "  [" + " ".join([f"{N[i,j]:8.4f}" for j in range(N.shape[1])]) + "]"
        lines.append(row_str)
    lines.append("")
    
    lines.append("7.4. Absorption probability matrix B = N · R")
    lines.append("B =")
    for i in range(B.shape[0]):
        row_str = "  [" + " ".join([f"{B[i,j]:8.6f}" for j in range(B.shape[1])]) + "]"
        lines.append(f"    (from state S_{transient_indices[i] + 1})")
        lines.append(row_str)
    lines.append("")
    
    lines.append("7.5. Interpretation")
    lines.append(f"From initial state S_{initial_state_idx + 1}:")
    for i, abs_idx in enumerate(absorbing_indices):
        if initial_state_idx in transient_indices:
            prob = B[transient_indices.index(initial_state_idx), i]
        else:
            prob = 1.0 if abs_idx == initial_state_idx else 0.0
        lines.append(f"  - Absorption probability in S_{abs_idx + 1}: {prob:.6f} ({prob*100:.2f}%)")
    lines.append("")
    lines.append(f"Sum of probabilities: {sum(absorption_probs.values()):.6f}")
    lines.append("")
    lines.append("=" * 80)
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"\nDetailed analysis appended to {output_file}")
    
    return {
        'absorption_probabilities': absorption_probs,
        'fundamental_matrix': N,
        'B': B,
        'transient_indices': transient_indices,
        'absorbing_indices': absorbing_indices
    }


if __name__ == "__main__":
    analyze_absorbing_states()

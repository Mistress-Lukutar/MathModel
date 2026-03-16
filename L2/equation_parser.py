"""
Parser for L1 exported equations.
Reads equations from Output/L1_equations.txt and converts to matrix form.

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-16
"""

import os
import re
import numpy as np


class EquationParser:
    """
    Parser for Kolmogorov equations exported from L1.
    
    Parses text file with format:
        dP_i/dt=coeff*P_j+coeff*P_k+...
    """
    
    def __init__(self, equations_path=None):
        """
        Initialize parser.
        
        Args:
            equations_path (str): Path to L1_equations.txt. If None, uses default.
        """
        if equations_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(script_dir)
            equations_path = os.path.join(root_dir, 'Output', 'L1_equations.txt')
        
        self.equations_path = equations_path
        self.n_states = 0
        self.Q = None
        self.initial_state = None
        self.absorbing_states = []
        self.equations = []
        
    def parse(self):
        """
        Parse the equations file.
        
        Returns:
            dict: Parsed data containing 'Q', 'initial_state', 'n_states', 'absorbing_states'
        """
        if not os.path.exists(self.equations_path):
            raise FileNotFoundError(f"Equations file not found: {self.equations_path}")
        
        with open(self.equations_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse metadata
        self._parse_metadata(content)
        
        # Parse equations
        self._parse_equations(content)
        
        # Parse initial conditions
        self._parse_initial_conditions(content)
        
        # Build matrix Q from equations
        self._build_matrix()
        
        return {
            'Q': self.Q,
            'initial_state': self.initial_state,
            'n_states': self.n_states,
            'absorbing_states': self.absorbing_states,
            'equations': self.equations
        }
    
    def _parse_metadata(self, content):
        """Parse metadata from comments."""
        # Parse number of states
        match = re.search(r'# States:\s*(\d+)', content)
        if match:
            self.n_states = int(match.group(1))
        
        # Parse absorbing states
        match = re.search(r'# Absorbing states:\s*\[(.*?)\]', content)
        if match:
            states_str = match.group(1)
            if states_str.strip():
                self.absorbing_states = [int(s.strip()) - 1 for s in states_str.split(',')]
        
        # Parse matrix Q from comments (lines starting with # containing matrix data)
        # Pattern: lines like "#  -0.5000   0.0000   0.1000 ..."
        q_lines = []
        lines = content.split('\n')
        in_matrix = False
        for line in lines:
            if 'Matrix Q' in line:
                in_matrix = True
                continue
            if in_matrix and line.startswith('#'):
                # Extract numbers from comment line
                # Remove leading "#" and spaces
                data_line = line[1:].strip()
                # Check if line contains numeric data
                if re.match(r'^[\s\d\.\-]+$', data_line):
                    try:
                        values = [float(v) for v in data_line.split()]
                        if len(values) > 0:
                            q_lines.append(values)
                    except ValueError:
                        pass
            elif in_matrix and not line.startswith('#'):
                break
        
        if q_lines and self.n_states == 0:
            self.n_states = len(q_lines)
        
        if q_lines:
            self.Q = np.array(q_lines)
            if self.Q.shape[0] != self.n_states:
                # Adjust to expected size
                self.Q = np.zeros((self.n_states, self.n_states))
                for i, row in enumerate(q_lines[:self.n_states]):
                    for j, v in enumerate(row[:self.n_states]):
                        self.Q[i, j] = v
    
    def _parse_equations(self, content):
        """Parse differential equations."""
        # Find equations section
        eq_pattern = r'dP_(\d+)/dt=([\d\.+\-*/P_]+)'
        matches = re.findall(eq_pattern, content)
        
        self.equations = []
        for state_num, eq_body in matches:
            i = int(state_num) - 1
            terms = self._parse_equation_body(eq_body, i)
            self.equations.append({
                'state': i,
                'terms': terms
            })
        
        # Update n_states if not set from metadata
        if self.n_states == 0 and self.equations:
            self.n_states = max(eq['state'] for eq in self.equations) + 1
    
    def _parse_equation_body(self, body, target_state):
        """
        Parse equation body like '0.05*P_3+0.20*P_4-0.70*P_1'.
        
        Returns:
            list: List of (source_state, coefficient) tuples
        """
        terms = []
        # Split by + or -, keeping the sign
        tokens = re.findall(r'([+-]?)(\d+\.?\d*)\*P_(\d+)', body)
        
        for sign, coeff, src_state in tokens:
            coef = float(coeff)
            if sign == '-':
                coef = -coef
            src = int(src_state) - 1
            terms.append((src, coef))
        
        return terms
    
    def _parse_initial_conditions(self, content):
        """Parse initial conditions P_i(0)=value."""
        ic_pattern = r'P_(\d+)\(0\)=([\d.]+)'
        matches = re.findall(ic_pattern, content)
        
        if matches:
            max_state = max(int(m[0]) for m in matches)
            self.initial_state = np.zeros(max_state)
            for state_str, val_str in matches:
                i = int(state_str) - 1
                self.initial_state[i] = float(val_str)
    
    def _build_matrix(self):
        """Build intensity matrix Q from parsed equations."""
        if self.Q is not None:
            return  # Already parsed from comments
        
        if self.n_states == 0:
            raise ValueError("Number of states not determined")
        
        self.Q = np.zeros((self.n_states, self.n_states))
        
        for eq in self.equations:
            i = eq['state']
            for src, coef in eq['terms']:
                if src == i:
                    # Diagonal term (outgoing rate)
                    self.Q[i, i] = coef
                else:
                    # Off-diagonal term (incoming from src to i)
                    self.Q[src, i] = coef


def load_from_L1(equations_path=None):
    """
    Convenience function to load equations from L1 export.
    
    Args:
        equations_path (str): Path to L1_equations.txt
        
    Returns:
        dict: Parsed data
    """
    parser = EquationParser(equations_path)
    return parser.parse()


if __name__ == "__main__":
    # Test parsing
    try:
        data = load_from_L1()
        print(f"Parsed {data['n_states']} states")
        print(f"Matrix Q:\n{data['Q']}")
        print(f"Initial state: {data['initial_state']}")
        print(f"Absorbing states: {data['absorbing_states']}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run L1 first to generate equations file.")

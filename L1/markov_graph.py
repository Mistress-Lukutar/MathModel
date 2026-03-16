"""
Markov Chain State Graph Generator using Graphviz with neato layout.
Reads configuration from root config.json.

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-16
"""

import json
import os
import subprocess
import tempfile


def load_config(config_path=None):
    """Load L1 configuration from root config.json."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        config_path = os.path.join(root_dir, 'config.json')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = json.load(f)
    
    return full_config['L1']


def detect_absorbing_states(config):
    """Detect absorbing states from transitions."""
    n = config['n_states']
    has_outgoing = set(tr['from'] for tr in config['transitions'])
    return [i for i in range(1, n + 1) if i not in has_outgoing]


def generate_dot_source(config):
    """Generate DOT source code for neato layout."""
    absorbing = detect_absorbing_states(config)
    
    lines = []
    lines.append('digraph MarkovChain {')
    lines.append('    graph [')
    lines.append('        bgcolor=white,')
    lines.append('        overlap=scalexy,')
    lines.append('        splines=true,')
    lines.append('        nodesep=1.0,')
    lines.append('        overlap=false,')
    lines.append('        ranksep=2.0,')
    lines.append('        fontname="Arial",')
    lines.append('        fontsize=28,')
    lines.append('        dpi=300,')
    lines.append('        pad=0.5')
    lines.append('    ];')
    lines.append('    node [')
    lines.append('        shape=circle,')
    lines.append('        style=filled,')
    lines.append('        width=1.0,')
    lines.append('        height=1.0,')
    lines.append('        fontsize=20,')
    lines.append('        fontname="Arial Bold",')
    lines.append('        penwidth=1')
    lines.append('    ];')
    lines.append('    edge [')
    lines.append('        penwidth=1,')
    lines.append('        arrowsize=1.2,')
    lines.append('        color=dimgray')
    lines.append('    ];')
    lines.append('')
    
    # Add nodes with colors
    for node in range(1, config['n_states'] + 1):
        is_absorbing = node in absorbing
        fillcolor = 'lightcoral' if is_absorbing else 'lightblue'
        color = 'darkred' if is_absorbing else 'navy'
        lines.append(f'    {node} [fillcolor="{fillcolor}", color="{color}"];')
    
    lines.append('')
    
    # Add edges with labels 
    for tr in config['transitions']:
        fro = tr['from']
        to = tr['to']
        rate = tr['rate']
        weight = max(1, int(rate * 15))
        length = max(1.5, 4.0 - rate * 3)
        
        # HTML-like label
        label_html = (f'<<TABLE BGCOLOR="white" BORDER="0" CELLSPACING="0" CELLPADDING="0">'
                      f'<TR><TD><FONT FACE="Arial Bold" POINT-SIZE="11" '
                      f'COLOR="black">{rate}</FONT></TD></TR></TABLE>>')
    
        lines.append(f'    {fro} -> {to} [label={label_html}, weight={weight}, len={length:.1f}];')

    # Add legend
    lines.append('')
    lines.append('    // Legend')
    lines.append('    legend [shape=plaintext, fontsize=13, fontname="Arial", '
                'style="filled", fillcolor="white", color="white", margin="0", label=<')
    lines.append('        <TABLE BGCOLOR="white" BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">')
    lines.append('            <TR><TD BGCOLOR="lightblue" WIDTH="25" HEIGHT="25"></TD><TD>Transient states</TD></TR>')
    lines.append('            <TR><TD BGCOLOR="lightcoral" WIDTH="25" HEIGHT="25"></TD><TD>Absorbing states</TD></TR>')
    lines.append('        </TABLE>>];')
    
    lines.append('}')
    
    return '\n'.join(lines)


def plot_markov_graph(config=None, output_file=None):
    """
    Generate Markov chain graph using Graphviz neato (spring layout).
    Nodes are positioned based on connection strength.
    """
    if config is None:
        config = load_config()
    
    if output_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        output_dir = os.path.join(root_dir, 'Output')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'L1_markov_graph.png')
    
    dot_source = generate_dot_source(config)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dot', delete=False) as f:
        f.write(dot_source)
        dot_file = f.name
    
    try:
        result = subprocess.run(
            ['neato', '-Tpng', '-o', output_file, dot_file],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Graph saved to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"stderr: {e.stderr}")
    except FileNotFoundError:
        print("Error: 'neato' not found. Make sure Graphviz is installed.")
    finally:
        os.unlink(dot_file)


if __name__ == "__main__":
    plot_markov_graph()

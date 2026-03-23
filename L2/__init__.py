"""
L2: Operator Method Solver for Kolmogorov Equations.

This module provides analytical solution of linear ODE systems
using Laplace transform (operator method).

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-16
"""

"""
L2: Operator Method Solver for Kolmogorov Equations.

This module provides analytical solution of linear ODE systems
using Laplace transform (operator method).

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-16
"""

from .equation_parser import EquationParser, load_from_L1
from .operator_solver import OperatorSolver, solve_from_L1
from .comparison import L2L1Comparator, compare_solutions
from .report_generator import L2ReportGenerator, generate_report

__all__ = [
    'EquationParser',
    'load_from_L1',
    'OperatorSolver',
    'solve_from_L1',
    'L2L1Comparator',
    'compare_solutions',
    'L2ReportGenerator',
    'generate_report',
]

__version__ = '1.0'
__author__ = 'Mistress-Lukutar'

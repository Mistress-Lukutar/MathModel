"""
L3: Numerical Solution of Linear Differential Equations.

This module provides numerical solution of linear ODE systems
using the Modified Euler method (Heun's method / predictor-corrector).

Variant 8: Modified Euler Method (2nd order accuracy)

Author: Mistress-Lukutar
Version: 1.0
Date: 2026-03-19
"""

from .modified_euler import ModifiedEulerSolver
from .step_analysis import StepConvergenceAnalyzer
from .comparison import L3L2Comparator
from .report_generator import L3ReportGenerator, generate_report

__all__ = [
    'ModifiedEulerSolver',
    'StepConvergenceAnalyzer',
    'L3L2Comparator',
    'L3ReportGenerator',
    'generate_report',
]

__version__ = '1.0'
__author__ = 'Mistress-Lukutar'

"""検算用のベンチマーク：Ishigami関数と、その Sobol 指数の理論値

Y = sin(x1) + a·sin²(x2) + b·x3⁴·sin(x1)、x1〜x3 は独立に一様分布 U(−π, π)
理論値の式は Ishigami & Homma (1990) による。
"""

import math

import numpy as np

from .parameters import Parameter

ISHIGAMI_A = 7.0
ISHIGAMI_B = 0.1
ISHIGAMI_PARAMETER_NAMES = ("x1", "x2", "x3")


def ishigami_parameters() -> list[Parameter]:
    return [
        Parameter(name=name, distribution="uniform", minimum=-math.pi, maximum=math.pi)
        for name in ISHIGAMI_PARAMETER_NAMES
    ]


def ishigami(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x1, x2, x3 = (inputs[name] for name in ISHIGAMI_PARAMETER_NAMES)
    return np.sin(x1) + ISHIGAMI_A * np.sin(x2) ** 2 + ISHIGAMI_B * x3**4 * np.sin(x1)


def ishigami_analytic_indices() -> tuple[np.ndarray, np.ndarray]:
    """(一次効果, 総合効果) の理論値"""
    a, b, pi = ISHIGAMI_A, ISHIGAMI_B, math.pi
    total_variance = a**2 / 8 + b * pi**4 / 5 + b**2 * pi**8 / 18 + 0.5
    variance_x1 = 0.5 * (1 + b * pi**4 / 5) ** 2
    variance_x2 = a**2 / 8
    variance_x1_x3 = b**2 * pi**8 * (1 / 18 - 1 / 50)
    first_order = np.array([variance_x1, variance_x2, 0.0]) / total_variance
    total_order = np.array([variance_x1 + variance_x1_x3, variance_x2, variance_x1_x3]) / total_variance
    return first_order, total_order

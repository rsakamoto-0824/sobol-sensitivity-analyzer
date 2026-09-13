"""Sobol指数の推定式とブートストラップ信頼区間

一次効果 S1: Saltelli et al. (2010) の推定式   V_i  ≈ mean( f(B) × (f(A_B^i) − f(A)) )
総合効果 ST: Jansen (1999) の推定式            V_Ti ≈ mean( (f(A) − f(A_B^i))² ) / 2
出力は全体の平均を引いて中心化し、f(A) と f(B) を合わせた標本分散で割る（SALib 1.5.2 の sobol.analyze と同じ処理）。
"""

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from .errors import SobolInputError

CONVERGENCE_STEP_COUNT = 3
MIN_CONVERGENCE_SAMPLES = 4


@dataclass(frozen=True)
class BlockOutputs:
    """1つの出力について、行列 A・B・A_B^(i) ごとに並べたモデル出力"""

    values_a: np.ndarray  # 形 (N,)
    values_b: np.ndarray  # 形 (N,)
    values_ab: np.ndarray  # 形 (d, N)。i 行目が A_B^(i+1) の出力

    @property
    def base_sample_count(self) -> int:
        return self.values_a.shape[0]

    def subset(self, sample_positions: np.ndarray) -> "BlockOutputs":
        return BlockOutputs(
            self.values_a[sample_positions], self.values_b[sample_positions], self.values_ab[:, sample_positions]
        )


@dataclass(frozen=True)
class SobolIndices:
    parameter_names: tuple[str, ...]
    base_sample_count: int
    confidence_level: float
    first_order: np.ndarray
    first_order_conf: np.ndarray  # 信頼区間の半幅（推定値 ± この値）
    total_order: np.ndarray
    total_order_conf: np.ndarray


@dataclass(frozen=True)
class ConvergenceStep:
    base_sample_count: int
    first_order: np.ndarray
    total_order: np.ndarray


def _estimate(outputs: BlockOutputs) -> tuple[np.ndarray, np.ndarray]:
    """一次効果と総合効果の推定値。出力の分散が 0 のときは NaN を返す"""
    dimension = outputs.values_ab.shape[0]
    # 一次効果の推定式は出力に一定値を足すと値が変わり、平均が大きい出力（寸法など）ほど誤差が増える。
    # そのため全体の平均を引いてから計算する
    center = np.mean(np.concatenate([outputs.values_a, outputs.values_b, outputs.values_ab.ravel()]))
    values_a = outputs.values_a - center
    values_b = outputs.values_b - center
    values_ab = outputs.values_ab - center

    all_values = np.concatenate([values_a, values_b])
    if np.ptp(all_values) == 0:
        return np.full(dimension, np.nan), np.full(dimension, np.nan)
    variance = np.var(all_values)
    difference = values_ab - values_a
    first_order = np.mean(values_b * difference, axis=1) / variance
    total_order = 0.5 * np.mean(difference**2, axis=1) / variance
    return first_order, total_order


def compute_sobol_indices(
    parameter_names: tuple[str, ...],
    outputs: BlockOutputs,
    bootstrap_count: int,
    confidence_level: float,
    seed: int | None = None,
) -> SobolIndices:
    if outputs.values_ab.shape[0] != len(parameter_names):
        raise ValueError("入力名の数と A_B 行列の数が一致しません")
    first_order, total_order = _estimate(outputs)
    if np.isnan(first_order).any():
        raise SobolInputError("出力がすべて同じ値のため、分散が 0 になり感度を計算できません")

    # サンプル番号 j を重複ありで選び直し（A・B・A_B で同じ j を使う）、推定値のばらつきを見る
    rng = np.random.default_rng(seed)
    sample_count = outputs.base_sample_count
    bootstrap_first = np.empty((bootstrap_count, len(parameter_names)))
    bootstrap_total = np.empty((bootstrap_count, len(parameter_names)))
    for trial in range(bootstrap_count):
        positions = rng.integers(0, sample_count, sample_count)
        bootstrap_first[trial], bootstrap_total[trial] = _estimate(outputs.subset(positions))

    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2)
    return SobolIndices(
        parameter_names=tuple(parameter_names),
        base_sample_count=sample_count,
        confidence_level=confidence_level,
        first_order=first_order,
        first_order_conf=z_value * np.nanstd(bootstrap_first, axis=0, ddof=1),
        total_order=total_order,
        total_order_conf=z_value * np.nanstd(bootstrap_total, axis=0, ddof=1),
    )


def compute_convergence(outputs: BlockOutputs) -> list[ConvergenceStep]:
    """先頭 N/4、N/2、N 個のサンプルで推定し直す。Sobol列は先頭の2のべき乗個でも均一なので、この確認に使える"""
    steps = []
    for shift in range(CONVERGENCE_STEP_COUNT - 1, -1, -1):
        count = outputs.base_sample_count >> shift
        if count < MIN_CONVERGENCE_SAMPLES:
            continue
        first_order, total_order = _estimate(outputs.subset(np.arange(count)))
        steps.append(ConvergenceStep(count, first_order, total_order))
    return steps

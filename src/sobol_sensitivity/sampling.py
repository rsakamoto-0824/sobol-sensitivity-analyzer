"""Saltelliサンプリング：行列 A・B と、列を1本だけ入れ替えた行列 A_B^(i) を作る"""

from dataclasses import dataclass

import numpy as np

from .errors import SobolInputError
from .estimators import BlockOutputs
from .parameters import Parameter, validate_parameter_list

SAMPLER_CHOICES = ("auto", "sobol", "random")
MIN_BASE_SAMPLES = 16
BLOCK_A = "A"
BLOCK_B = "B"
AB_BLOCK_PREFIX = "AB_"


@dataclass(frozen=True)
class SaltelliDesign:
    """Sobol指数の推定に使う入力サンプル一式（N は基本サンプル数、d は入力の数）"""

    parameter_names: tuple[str, ...]
    matrix_a: np.ndarray  # 形 (N, d)
    matrix_b: np.ndarray  # 形 (N, d)
    matrices_ab: tuple[np.ndarray, ...]  # d 個。i 番目は A の i 列目を B の i 列目に入れ替えたもの
    sampler_used: str

    @property
    def base_sample_count(self) -> int:
        return self.matrix_a.shape[0]

    @property
    def total_run_count(self) -> int:
        return self.base_sample_count * (len(self.parameter_names) + 2)

    def block_names(self) -> list[str]:
        return [BLOCK_A, BLOCK_B, *(ab_block_name(index) for index in range(len(self.parameter_names)))]

    def stacked_inputs(self) -> np.ndarray:
        """A, B, A_B^(1), …, A_B^(d) の順に縦へ並べた入力（形 (N(d+2), d)）"""
        return np.vstack([self.matrix_a, self.matrix_b, *self.matrices_ab])


def ab_block_name(parameter_index: int) -> str:
    return f"{AB_BLOCK_PREFIX}{parameter_index + 1}"


def validate_base_sample_count(count: int) -> None:
    is_power_of_two = count > 0 and (count & (count - 1)) == 0
    if count < MIN_BASE_SAMPLES or not is_power_of_two:
        raise SobolInputError(
            f"サンプル数 N は {MIN_BASE_SAMPLES} 以上の2のべき乗にしてください（例: 1024, 2048, 4096）。指定値: {count}"
        )


def _load_scipy_qmc():
    """SciPy があれば準乱数モジュールを返し、無ければ None を返す（テストで差し替えるため関数にしている）"""
    try:
        from scipy.stats import qmc
    except ImportError:
        return None
    return qmc


def generate_unit_samples(count: int, dimension: int, sampler: str, seed: int | None) -> tuple[np.ndarray, str, list[str]]:
    """[0, 1) の一様な点を count 行 × dimension 列で作る。戻り値は (点, 実際に使った方法, 利用者へのお知らせ)"""
    if sampler not in SAMPLER_CHOICES:
        raise SobolInputError(f"sampler は {' / '.join(SAMPLER_CHOICES)} のいずれかにしてください（指定値: {sampler}）")
    validate_base_sample_count(count)

    qmc = _load_scipy_qmc() if sampler in ("auto", "sobol") else None
    if sampler == "sobol" and qmc is None:
        raise SobolInputError("--sampler sobol には SciPy が必要です。SciPy を入れるか、--sampler random を指定してください")
    if qmc is not None:
        engine = qmc.Sobol(d=dimension, scramble=True, rng=np.random.default_rng(seed))
        return engine.random_base2(m=count.bit_length() - 1), "sobol", []

    notes = []
    if sampler == "auto":
        notes.append("SciPy が見つからないため、Sobol列ではなく通常の乱数でサンプルを作りました。N を大きめにしてください。")
    return np.random.default_rng(seed).random((count, dimension)), "random", notes


def build_saltelli_design(
    parameters: list[Parameter], base_sample_count: int, sampler: str = "auto", seed: int | None = None
) -> tuple[SaltelliDesign, list[str]]:
    validate_parameter_list(parameters)
    dimension = len(parameters)
    unit_samples, sampler_used, notes = generate_unit_samples(base_sample_count, 2 * dimension, sampler, seed)
    matrix_a = _transform_columns(parameters, unit_samples[:, :dimension])
    matrix_b = _transform_columns(parameters, unit_samples[:, dimension:])
    matrices_ab = []
    for index in range(dimension):
        swapped = matrix_a.copy()
        swapped[:, index] = matrix_b[:, index]
        matrices_ab.append(swapped)
    design = SaltelliDesign(
        parameter_names=tuple(parameter.name for parameter in parameters),
        matrix_a=matrix_a,
        matrix_b=matrix_b,
        matrices_ab=tuple(matrices_ab),
        sampler_used=sampler_used,
    )
    return design, notes


def _transform_columns(parameters: list[Parameter], unit_matrix: np.ndarray) -> np.ndarray:
    return np.column_stack([parameter.transform(unit_matrix[:, index]) for index, parameter in enumerate(parameters)])


def split_stacked_outputs(stacked_values: np.ndarray, base_sample_count: int, dimension: int) -> BlockOutputs:
    """stacked_inputs() と同じ順（A, B, A_B^(1), …）に並んだ出力を、行列ごとに分ける"""
    count = base_sample_count
    return BlockOutputs(
        values_a=stacked_values[:count],
        values_b=stacked_values[count : 2 * count],
        values_ab=stacked_values[2 * count :].reshape(dimension, count),
    )

"""推定式の正しさを、理論値が分かっているモデルで確かめる"""

import numpy as np
import pytest

from sobol_sensitivity.benchmark import ishigami, ishigami_analytic_indices, ishigami_parameters
from sobol_sensitivity.errors import SobolInputError
from sobol_sensitivity.estimators import compute_convergence, compute_sobol_indices
from sobol_sensitivity.model_runner import DEFAULT_OUTPUT_NAME, evaluate_model
from sobol_sensitivity.parameters import Parameter
from sobol_sensitivity.sampling import build_saltelli_design, split_stacked_outputs

BASE_SAMPLES = 4096
BOOTSTRAP_COUNT = 200
CONFIDENCE_LEVEL = 0.95
SEED = 2024
TOLERANCE = 0.02
RANDOM_SAMPLER_TOLERANCE = 0.05


def _design_and_outputs(model, parameters, sampler="sobol", base_samples=BASE_SAMPLES):
    design, _ = build_saltelli_design(parameters, base_samples, sampler, SEED)
    values = evaluate_model(model, design.parameter_names, design.stacked_inputs())[DEFAULT_OUTPUT_NAME]
    return design, split_stacked_outputs(values, design.base_sample_count, len(parameters))


def _indices(model, parameters, sampler="sobol", base_samples=BASE_SAMPLES):
    design, outputs = _design_and_outputs(model, parameters, sampler, base_samples)
    return compute_sobol_indices(design.parameter_names, outputs, BOOTSTRAP_COUNT, CONFIDENCE_LEVEL, SEED)


def _uniform(name, minimum, maximum):
    return Parameter(name=name, distribution="uniform", minimum=minimum, maximum=maximum)


def test_additive_model_matches_hand_calculation():
    # Y = x1 + 2·x2、x ~ U(0, 1) → S1 = (1/12)/(5/12) = 0.2、S2 = 0.8、交互作用なし
    indices = _indices(lambda x: x["x1"] + 2 * x["x2"], [_uniform("x1", 0, 1), _uniform("x2", 0, 1)])
    np.testing.assert_allclose(indices.first_order, [0.2, 0.8], atol=TOLERANCE)
    np.testing.assert_allclose(indices.total_order, [0.2, 0.8], atol=TOLERANCE)


def test_product_model_has_only_interaction():
    # Y = x1·x2、x ~ U(−1, 1) → S1 = S2 = 0、ST1 = ST2 = 1
    indices = _indices(lambda x: x["x1"] * x["x2"], [_uniform("x1", -1, 1), _uniform("x2", -1, 1)])
    np.testing.assert_allclose(indices.first_order, [0.0, 0.0], atol=TOLERANCE)
    np.testing.assert_allclose(indices.total_order, [1.0, 1.0], atol=TOLERANCE)


def test_normal_and_triangular_inputs_match_variance_ratio():
    # Y = u + 2n + t、u ~ U(0, 1)（分散 1/12）、n ~ N(0, 1)（2n の分散 4）、t ~ 三角(−1, 0, 1)（分散 1/6）
    parameters = [
        _uniform("u", 0, 1),
        Parameter(name="n", distribution="normal", mean=0.0, std=1.0),
        Parameter(name="t", distribution="triangular", minimum=-1.0, maximum=1.0, mode=0.0),
    ]
    variances = np.array([1 / 12, 4.0, 1 / 6])
    indices = _indices(lambda x: x["u"] + 2 * x["n"] + x["t"], parameters)
    np.testing.assert_allclose(indices.first_order, variances / variances.sum(), atol=TOLERANCE)
    np.testing.assert_allclose(indices.total_order, variances / variances.sum(), atol=TOLERANCE)


@pytest.mark.parametrize(
    ("sampler", "base_samples", "tolerance"),
    [("sobol", BASE_SAMPLES, TOLERANCE), ("random", 4 * BASE_SAMPLES, RANDOM_SAMPLER_TOLERANCE)],
)
def test_ishigami_matches_analytic_values(sampler, base_samples, tolerance):
    indices = _indices(ishigami, ishigami_parameters(), sampler, base_samples)
    expected_first, expected_total = ishigami_analytic_indices()
    np.testing.assert_allclose(indices.first_order, expected_first, atol=tolerance)
    np.testing.assert_allclose(indices.total_order, expected_total, atol=tolerance)


def test_adding_constant_to_output_does_not_change_indices():
    # 寸法のように平均が大きく、ばらつきが小さい出力でも結果が変わらないこと
    parameters = [_uniform("x1", 0, 1), _uniform("x2", 0, 1)]
    base = _indices(lambda x: x["x1"] + 2 * x["x2"], parameters, base_samples=256)
    shifted = _indices(lambda x: 1000.0 + x["x1"] + 2 * x["x2"], parameters, base_samples=256)
    np.testing.assert_allclose(shifted.first_order, base.first_order, atol=1e-9)
    np.testing.assert_allclose(shifted.total_order, base.total_order, atol=1e-9)


def test_point_estimates_match_salib_when_installed():
    salib_sobol = pytest.importorskip("SALib.analyze.sobol")
    design, outputs = _design_and_outputs(ishigami, ishigami_parameters(), base_samples=1024)
    ours = compute_sobol_indices(design.parameter_names, outputs, BOOTSTRAP_COUNT, CONFIDENCE_LEVEL, SEED)
    # SALib は、サンプルごとに A, A_B^(1) … A_B^(d), B の順で並んだ出力を受け取る
    interleaved = np.column_stack([outputs.values_a, *outputs.values_ab, outputs.values_b]).ravel()
    dimension = len(design.parameter_names)
    problem = {"num_vars": dimension, "names": list(design.parameter_names), "bounds": [[-np.pi, np.pi]] * dimension}
    expected = salib_sobol.analyze(
        problem, interleaved, calc_second_order=False, num_resamples=BOOTSTRAP_COUNT, seed=SEED
    )
    np.testing.assert_allclose(ours.first_order, expected["S1"], atol=1e-10)
    np.testing.assert_allclose(ours.total_order, expected["ST"], atol=1e-10)


def test_confidence_intervals_are_positive_and_reproducible():
    design, outputs = _design_and_outputs(ishigami, ishigami_parameters(), base_samples=1024)
    first = compute_sobol_indices(design.parameter_names, outputs, BOOTSTRAP_COUNT, CONFIDENCE_LEVEL, SEED)
    second = compute_sobol_indices(design.parameter_names, outputs, BOOTSTRAP_COUNT, CONFIDENCE_LEVEL, SEED)
    assert np.all(first.first_order_conf > 0)
    assert np.all(first.total_order_conf > 0)
    np.testing.assert_array_equal(first.first_order_conf, second.first_order_conf)


def test_constant_output_is_rejected():
    parameters = [_uniform("x1", 0, 1), _uniform("x2", 0, 1)]
    design, outputs = _design_and_outputs(lambda x: np.ones_like(x["x1"]), parameters, base_samples=64)
    with pytest.raises(SobolInputError, match="分散が 0"):
        compute_sobol_indices(design.parameter_names, outputs, BOOTSTRAP_COUNT, CONFIDENCE_LEVEL, SEED)


def test_convergence_uses_quarter_half_and_full_samples():
    design, outputs = _design_and_outputs(ishigami, ishigami_parameters(), base_samples=1024)
    steps = compute_convergence(outputs)
    assert [step.base_sample_count for step in steps] == [256, 512, 1024]
    indices = compute_sobol_indices(design.parameter_names, outputs, BOOTSTRAP_COUNT, CONFIDENCE_LEVEL, SEED)
    np.testing.assert_allclose(steps[-1].first_order, indices.first_order)
    np.testing.assert_allclose(steps[-1].total_order, indices.total_order)

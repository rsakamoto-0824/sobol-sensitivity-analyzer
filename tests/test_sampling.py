"""Saltelliサンプリングのテスト"""

import numpy as np
import pytest

from sobol_sensitivity import sampling
from sobol_sensitivity.errors import SobolInputError
from sobol_sensitivity.parameters import Parameter
from sobol_sensitivity.sampling import build_saltelli_design, split_stacked_outputs

PARAMETERS = [
    Parameter(name="a", distribution="uniform", minimum=0.0, maximum=1.0),
    Parameter(name="b", distribution="normal", mean=5.0, std=2.0),
    Parameter(name="c", distribution="triangular", minimum=-1.0, maximum=1.0, mode=0.5),
]
BASE_SAMPLES = 64
SEED = 7


def test_ab_matrices_take_only_one_column_from_b():
    design, _ = build_saltelli_design(PARAMETERS, BASE_SAMPLES, "sobol", SEED)
    assert design.sampler_used == "sobol"
    for index, matrix_ab in enumerate(design.matrices_ab):
        for column in range(len(PARAMETERS)):
            source = design.matrix_b if column == index else design.matrix_a
            np.testing.assert_array_equal(matrix_ab[:, column], source[:, column])
    assert design.stacked_inputs().shape == (BASE_SAMPLES * 5, 3)
    assert design.total_run_count == BASE_SAMPLES * 5
    assert design.block_names() == ["A", "B", "AB_1", "AB_2", "AB_3"]


@pytest.mark.parametrize("count", [0, 8, 1000, 1025])
def test_base_sample_count_must_be_power_of_two(count):
    with pytest.raises(SobolInputError, match="2のべき乗"):
        build_saltelli_design(PARAMETERS, count, "auto", SEED)


def test_same_seed_gives_same_samples():
    first, _ = build_saltelli_design(PARAMETERS, BASE_SAMPLES, "auto", SEED)
    second, _ = build_saltelli_design(PARAMETERS, BASE_SAMPLES, "auto", SEED)
    np.testing.assert_array_equal(first.stacked_inputs(), second.stacked_inputs())


def test_auto_sampler_falls_back_to_random_without_scipy(monkeypatch):
    monkeypatch.setattr(sampling, "_load_scipy_qmc", lambda: None)
    design, notes = build_saltelli_design(PARAMETERS, BASE_SAMPLES, "auto", SEED)
    assert design.sampler_used == "random"
    assert any("SciPy" in note for note in notes)


def test_sobol_sampler_requires_scipy(monkeypatch):
    monkeypatch.setattr(sampling, "_load_scipy_qmc", lambda: None)
    with pytest.raises(SobolInputError, match="SciPy が必要"):
        build_saltelli_design(PARAMETERS, BASE_SAMPLES, "sobol", SEED)


def test_split_stacked_outputs_follows_block_order():
    dimension = 3
    stacked = np.arange(BASE_SAMPLES * (dimension + 2), dtype=float)
    outputs = split_stacked_outputs(stacked, BASE_SAMPLES, dimension)
    np.testing.assert_array_equal(outputs.values_a, stacked[:BASE_SAMPLES])
    np.testing.assert_array_equal(outputs.values_b, stacked[BASE_SAMPLES : 2 * BASE_SAMPLES])
    np.testing.assert_array_equal(outputs.values_ab[1], stacked[3 * BASE_SAMPLES : 4 * BASE_SAMPLES])

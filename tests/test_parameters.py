"""入力パラメータ定義CSVの読み込みと、分布への変換のテスト"""

import numpy as np
import pytest

from sobol_sensitivity.errors import SobolInputError
from sobol_sensitivity.parameters import Parameter, read_parameters_csv

VALID_CSV = """name,distribution,min,max,mean,std,mode,memo
width,uniform,1,3,,,,メモ欄は読み飛ばす
dose,normal,,,20,0.5,,
focus,triangular,-0.1,0.1,,,0.02,
"""
HEADER_LINE = "name,distribution,min,max,mean,std,mode\n"
EVENLY_SPACED_COUNT = 100_000
UNIT_VALUES = (np.arange(EVENLY_SPACED_COUNT) + 0.5) / EVENLY_SPACED_COUNT


def _write_csv(tmp_path, text, encoding="utf-8-sig"):
    path = tmp_path / "parameters.csv"
    path.write_text(text, encoding=encoding)
    return path


def test_reads_all_distributions_and_ignores_extra_columns(tmp_path):
    parameters = read_parameters_csv(_write_csv(tmp_path, VALID_CSV))
    assert parameters == [
        Parameter(name="width", distribution="uniform", minimum=1.0, maximum=3.0),
        Parameter(name="dose", distribution="normal", mean=20.0, std=0.5),
        Parameter(name="focus", distribution="triangular", minimum=-0.1, maximum=0.1, mode=0.02),
    ]


def test_reads_shift_jis_csv_with_japanese_names(tmp_path):
    text = "name,distribution,min,max\n線幅,uniform,0,1\n膜厚,uniform,2,4\n"
    parameters = read_parameters_csv(_write_csv(tmp_path, text, encoding="cp932"))
    assert [parameter.name for parameter in parameters] == ["線幅", "膜厚"]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("a,gamma,0,1,,,\nb,uniform,0,1,,,", "distribution は"),
        ("a,uniform,1,1,,,\nb,uniform,0,1,,,", "min は max より小さく"),
        ("a,normal,,,0,0,\nb,uniform,0,1,,,", "std は 0 より大きく"),
        ("a,triangular,0,1,,,2\nb,uniform,0,1,,,", "mode は min 以上"),
        ("a,uniform,0,,,,\nb,uniform,0,1,,,", "max 列に値が必要"),
        ("a,uniform,0,abc,,,\nb,uniform,0,1,,,", "数値として読めません"),
        ("a,uniform,0,1,,,\na,uniform,0,1,,,", "重複"),
        ("run_id,uniform,0,1,,,\nb,uniform,0,1,,,", "入力名に"),
        ("a,uniform,0,1,,,", "2 個以上"),
    ],
)
def test_rejects_invalid_definitions(tmp_path, rows, message):
    path = _write_csv(tmp_path, HEADER_LINE + rows + "\n")
    with pytest.raises(SobolInputError, match=message):
        read_parameters_csv(path)


def test_rejects_missing_required_column(tmp_path):
    with pytest.raises(SobolInputError, match="必須の列"):
        read_parameters_csv(_write_csv(tmp_path, "name,min,max\na,0,1\nb,0,1\n"))


def test_rejects_missing_file(tmp_path):
    with pytest.raises(SobolInputError, match="見つかりません"):
        read_parameters_csv(tmp_path / "missing.csv")


def test_uniform_transform_stays_in_range():
    values = Parameter(name="a", distribution="uniform", minimum=-2.0, maximum=6.0).transform(UNIT_VALUES)
    assert values.min() >= -2.0
    assert values.max() <= 6.0
    assert values.mean() == pytest.approx(2.0, abs=1e-3)


def test_normal_transform_matches_mean_and_std():
    values = Parameter(name="a", distribution="normal", mean=20.0, std=0.5).transform(UNIT_VALUES)
    assert values.mean() == pytest.approx(20.0, abs=1e-3)
    assert values.std() == pytest.approx(0.5, rel=1e-2)


def test_triangular_transform_matches_range_and_mean():
    parameter = Parameter(name="a", distribution="triangular", minimum=-1.0, maximum=3.0, mode=0.0)
    values = parameter.transform(UNIT_VALUES)
    assert values.min() >= -1.0
    assert values.max() <= 3.0
    assert values.mean() == pytest.approx((-1.0 + 3.0 + 0.0) / 3, abs=1e-3)

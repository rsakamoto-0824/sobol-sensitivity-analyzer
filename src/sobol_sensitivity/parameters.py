"""入力パラメータ定義CSVの読み込みと、[0, 1) の一様乱数から各分布の値への変換"""

from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np

from .csv_io import META_COLUMNS, parse_float, read_csv_table
from .errors import SobolInputError

REQUIRED_COLUMNS = ("name", "distribution")
# 分布ごとに必要な列。これ以外の列（メモ欄など）は読み飛ばす
REQUIRED_VALUES_BY_DISTRIBUTION = {
    "uniform": ("min", "max"),
    "normal": ("mean", "std"),
    "triangular": ("min", "max", "mode"),
}
COLUMN_TO_ATTRIBUTE = {"min": "minimum", "max": "maximum", "mean": "mean", "std": "std", "mode": "mode"}
MIN_PARAMETER_COUNT = 2
# 正規分布の逆関数に 0 や 1 を渡すと無限大になるため、この幅だけ内側に寄せる
PROBABILITY_EPSILON = 1e-12


@dataclass(frozen=True)
class Parameter:
    """1つの入力と、その入力が従う分布"""

    name: str
    distribution: str
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    std: float | None = None
    mode: float | None = None

    def validate(self, location: str) -> None:
        if self.distribution not in REQUIRED_VALUES_BY_DISTRIBUTION:
            choices = " / ".join(REQUIRED_VALUES_BY_DISTRIBUTION)
            raise SobolInputError(
                f"{location}: distribution は {choices} のいずれかにしてください（指定値: '{self.distribution}'）"
            )
        for column in REQUIRED_VALUES_BY_DISTRIBUTION[self.distribution]:
            if getattr(self, COLUMN_TO_ATTRIBUTE[column]) is None:
                raise SobolInputError(f"{location}: distribution が {self.distribution} のときは {column} の値が必要です")
        if self.distribution in ("uniform", "triangular") and not self.minimum < self.maximum:
            raise SobolInputError(f"{location}: min は max より小さくしてください（min={self.minimum}, max={self.maximum}）")
        if self.distribution == "normal" and not self.std > 0:
            raise SobolInputError(f"{location}: std は 0 より大きくしてください（std={self.std}）")
        if self.distribution == "triangular" and not self.minimum <= self.mode <= self.maximum:
            raise SobolInputError(f"{location}: mode は min 以上 max 以下にしてください（mode={self.mode}）")

    def transform(self, unit_values: np.ndarray) -> np.ndarray:
        """[0, 1) の一様乱数（1次元配列）を、この入力の分布に従う値へ変換する（逆関数法）"""
        if self.distribution == "uniform":
            return self.minimum + unit_values * (self.maximum - self.minimum)
        if self.distribution == "normal":
            normal = NormalDist(self.mean, self.std)
            clipped = np.clip(unit_values, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
            return np.fromiter((normal.inv_cdf(float(value)) for value in clipped), dtype=float, count=clipped.size)
        width = self.maximum - self.minimum
        mode_fraction = (self.mode - self.minimum) / width
        left_side = self.minimum + np.sqrt(unit_values * width * (self.mode - self.minimum))
        right_side = self.maximum - np.sqrt((1 - unit_values) * width * (self.maximum - self.mode))
        return np.where(unit_values < mode_fraction, left_side, right_side)


def read_parameters_csv(path: str | Path) -> list[Parameter]:
    table = read_csv_table(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in table.header]
    if missing_columns:
        raise SobolInputError(f"{table.path.name}: 必須の列がありません: {', '.join(missing_columns)}")
    parameters = [
        _parameter_from_row(row, f"{table.path.name} {line_number}行目") for line_number, row in table.rows
    ]
    validate_parameter_list(parameters)
    return parameters


def _parameter_from_row(row: dict[str, str], location: str) -> Parameter:
    name = row["name"]
    distribution = row["distribution"].lower()
    if not name:
        raise SobolInputError(f"{location}: name が空です")
    values = {}
    for column in REQUIRED_VALUES_BY_DISTRIBUTION.get(distribution, ()):
        text = row.get(column, "")
        if not text:
            raise SobolInputError(f"{location}: distribution が {distribution} のときは {column} 列に値が必要です")
        values[COLUMN_TO_ATTRIBUTE[column]] = parse_float(text, f"{location} の {column} 列")
    parameter = Parameter(name=name, distribution=distribution, **values)
    parameter.validate(location)
    return parameter


def validate_parameter_list(parameters: list[Parameter]) -> None:
    if len(parameters) < MIN_PARAMETER_COUNT:
        raise SobolInputError(f"入力は {MIN_PARAMETER_COUNT} 個以上定義してください（現在 {len(parameters)} 個）")
    names = [parameter.name for parameter in parameters]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SobolInputError(f"入力名が重複しています: {', '.join(duplicates)}")
    reserved = [name for name in names if name in META_COLUMNS]
    if reserved:
        raise SobolInputError(f"入力名に {', '.join(META_COLUMNS)} は使えません: {', '.join(reserved)}")

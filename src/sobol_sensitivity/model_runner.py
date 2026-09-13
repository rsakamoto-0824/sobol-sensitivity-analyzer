"""利用者が書いたPythonファイルからモデル関数を読み込み、全サンプルをまとめて評価する

モデル関数の約束:
    def model(inputs):   # inputs は {入力名: 1次元の numpy 配列} の辞書（全サンプル分）
        return 出力       # 出力が1つなら配列、複数なら {出力名: 配列} の辞書
"""

import importlib.util
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .csv_io import META_COLUMNS
from .errors import SobolInputError

DEFAULT_FUNCTION_NAME = "model"
DEFAULT_OUTPUT_NAME = "Y"
USER_MODULE_NAME = "sobol_user_model"


def load_model_function(path: str | Path, function_name: str = DEFAULT_FUNCTION_NAME) -> Callable:
    model_path = Path(path)
    if not model_path.is_file():
        raise SobolInputError(f"モデルファイルが見つかりません: {model_path}")
    if model_path.suffix != ".py":
        raise SobolInputError(f"モデルファイルには .py ファイルを指定してください: {model_path}")

    spec = importlib.util.spec_from_file_location(USER_MODULE_NAME, model_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise SobolInputError(
            f"モデルファイルの読み込み中にエラーが発生しました（{type(error).__name__}: {error}）: {model_path}"
        ) from error

    function = getattr(module, function_name, None)
    if not callable(function):
        raise SobolInputError(f"モデルファイルに関数 {function_name} が見つかりません: {model_path}")
    return function


def evaluate_model(function: Callable, parameter_names: tuple[str, ...], inputs: np.ndarray) -> dict[str, np.ndarray]:
    row_count = inputs.shape[0]
    # 利用者の関数が配列を書き換えても、保存するサンプルに影響しないようコピーを渡す
    named_inputs = {name: inputs[:, index].copy() for index, name in enumerate(parameter_names)}
    try:
        result = function(named_inputs)
    except Exception as error:
        raise SobolInputError(f"モデル関数の実行中にエラーが発生しました（{type(error).__name__}: {error}）") from error

    named_results = result if isinstance(result, dict) else {DEFAULT_OUTPUT_NAME: result}
    if not named_results:
        raise SobolInputError("モデル関数が出力を返していません")

    reserved_names = {*META_COLUMNS, *parameter_names}
    outputs = {}
    for raw_name, values in named_results.items():
        name = str(raw_name)
        if name in reserved_names:
            raise SobolInputError(f"出力名 '{name}' は入力名または管理用の列名（{', '.join(META_COLUMNS)}）と重複しています")
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as error:
            raise SobolInputError(f"出力 '{name}' を数値の配列に変換できません（{error}）") from error
        if array.shape != (row_count,):
            raise SobolInputError(
                f"出力 '{name}' の形が {array.shape} です。入力と同じ長さ {row_count} の1次元配列を返してください"
            )
        invalid_count = int(np.count_nonzero(~np.isfinite(array)))
        if invalid_count:
            raise SobolInputError(f"出力 '{name}' に NaN または無限大が {invalid_count} 個含まれています")
        outputs[name] = array
    return outputs

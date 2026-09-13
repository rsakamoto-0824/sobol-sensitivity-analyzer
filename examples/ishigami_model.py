"""モデル関数の例：Ishigami関数（検算用のベンチマーク）

モデル関数の書き方
  - 関数名は model にする（別の名前にする場合は --function で指定する）
  - 引数 inputs は {入力名: numpy 配列} の辞書。入力名はパラメータCSVの name 列と同じ
  - 全サンプル分の配列がまとめて渡されるので、numpy の配列計算で一度に計算する
  - 戻り値は入力と同じ長さの配列。出力が複数あるときは {出力名: 配列} の辞書で返す
"""

import numpy as np

COEFFICIENT_A = 7.0
COEFFICIENT_B = 0.1


def model(inputs):
    x1 = inputs["x1"]
    x2 = inputs["x2"]
    x3 = inputs["x3"]
    return np.sin(x1) + COEFFICIENT_A * np.sin(x2) ** 2 + COEFFICIENT_B * x3**4 * np.sin(x1)

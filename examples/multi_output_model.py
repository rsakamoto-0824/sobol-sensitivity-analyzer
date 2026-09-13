"""モデル関数の例：出力が2つあるモデル（動作確認用の架空の式）

  linear_output      = uniform_input + 2 × normal_input + triangular_input   … 交互作用なし
  interaction_output = uniform_input × triangular_input                      … 交互作用を通じて効く入力がある

normal_input は interaction_output の計算に使わないため、その出力に対する総合効果は 0 になる。
"""


def model(inputs):
    uniform_input = inputs["uniform_input"]
    normal_input = inputs["normal_input"]
    triangular_input = inputs["triangular_input"]
    return {
        "linear_output": uniform_input + 2 * normal_input + triangular_input,
        "interaction_output": uniform_input * triangular_input,
    }

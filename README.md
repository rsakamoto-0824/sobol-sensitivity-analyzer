# sobol-sensitivity-analyzer

シミュレーションや計算式の出力のばらつきが、どの入力からどれだけ来ているかを **Sobol感度解析** で数値化するコマンドラインツールです。
一次効果 **S1**（その入力だけの寄与）と総合効果 **ST**（交互作用を含めた寄与）を、信頼区間つきで計算します。

考え方は学習用資料 [docs/Sobol感度解析_概念と原理.pptx](docs/Sobol感度解析_概念と原理.pptx) を参照してください。

## できること

| コマンド | 用途 |
|---|---|
| `check` | Ishigami関数で計算し、理論値と一致するか確かめる（環境の動作確認） |
| `run` | モデルを Python の関数で書ける場合に、サンプル作成から指数の計算までまとめて行う |
| `sample` → `analyze` | Excel やシミュレータなど外部ツールで計算する場合に、入力サンプルCSVを作り、計算結果を読み込んで指数を求める |

- 入力の分布: 一様分布・正規分布・三角分布
- 出力が複数あるモデルにも対応
- SciPy が無くても動く（その場合は Sobol列の代わりに通常の乱数を使う）

## 必要な環境

| 項目 | バージョン | 備考 |
|---|---|---|
| Python | 3.14.5 以上 | `pyproject.toml` の `requires-python` と起動時のチェックで確認 |
| NumPy | 2.3.2 以上 | 必須 |
| SciPy | 1.16.1 以上 | 任意。Sobol列を使うため、あると同じ計算回数でも精度が上がる |
| pytest | 9.1 以上 | テストを実行する場合のみ |

- 動作確認済み: macOS、Python 3.14.6、NumPy 2.5.3、SciPy 1.18.1、pytest 9.1.1
- **Windows での動作は未確認です**（[issues.md](issues.md) 参照）
- 環境変数の設定は不要です

## セットアップ（Windows 11）

コマンドプロンプトで、このフォルダに移動してから実行します。

```bat
py -3.14 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python run_sobol.py check
```

最後に「判定: 合格」と表示されれば準備完了です。

- **SciPy をインストールできない場合**: `requirements.txt` から `scipy` の行を消して、もう一度 `pip install` してください。
- **pip がインターネットにつながらない場合**: プロキシ設定など社内の手順を、情報システム担当に確認してください。

macOS の場合は、1〜2行目を `python3.14 -m venv .venv` と `source .venv/bin/activate` に読み替えます。

## 使い方

### 1. 入力パラメータCSVを作る

Excel で作成し、CSV 形式で保存します（UTF-8・Shift_JIS のどちらでも読めます）。1行が1つの入力です。

| 列 | 内容 |
|---|---|
| `name` | 入力名（重複不可。`run_id` / `block` / `sample_index` は使えない） |
| `distribution` | `uniform` / `normal` / `triangular` |
| `min`, `max`, `mean`, `std`, `mode` | 分布に応じて必要な列だけ埋める（下表） |
| その他の列 | 読み飛ばすので、メモ欄として使える |

| distribution | 必要な列 | 意味 |
|---|---|---|
| `uniform` | `min`, `max` | min〜max の範囲で一様にばらつく |
| `normal` | `mean`, `std` | 平均 mean・標準偏差 std の正規分布 |
| `triangular` | `min`, `max`, `mode` | min〜max の範囲で、mode が最も起こりやすい三角分布 |

例: [examples/multi_output_parameters.csv](examples/multi_output_parameters.csv)

```csv
name,distribution,min,max,mean,std,mode,memo
uniform_input,uniform,0,1,,,,一様分布：min〜max
normal_input,normal,,,0,1,,正規分布：平均 mean・標準偏差 std
triangular_input,triangular,-1,1,,,0,三角分布：min〜max・最頻値 mode
```

### 2-A. モデルを Python で書ける場合（`run`）

モデル関数を `.py` ファイルに書きます。全サンプル分の配列がまとめて渡されるので、NumPy の配列計算で一度に計算します。

```python
import numpy as np

def model(inputs):
    # inputs は {入力名: numpy 配列} の辞書。入力名はパラメータCSVの name 列と同じ
    x1 = inputs["x1"]
    x2 = inputs["x2"]
    return x1 + 2 * np.sin(x2)          # 出力が1つなら配列を返す（出力名は Y になる）
    # return {"CD": ..., "LER": ...}    # 出力が複数なら {出力名: 配列} の辞書を返す
```

```bat
python run_sobol.py run --params examples\ishigami_parameters.csv --model examples\ishigami_model.py --n 4096 --out-dir results\ishigami
```

1行ずつしか計算できない関数を使う場合は、関数の中でループします。

```python
import numpy as np

def model(inputs):
    count = len(inputs["x1"])
    return np.array([my_simulation(inputs["x1"][i], inputs["x2"][i]) for i in range(count)])
```

### 2-B. 外部ツールで計算する場合（`sample` → `analyze`）

1. 入力サンプルCSVを作ります。

   ```bat
   python run_sobol.py sample --params my_parameters.csv --n 1024 --out work\samples.csv
   ```

2. `samples.csv` の各行の入力値で、外部ツールを使って出力を計算します。結果は次のどちらかの形で保存します。
   - **サンプルCSVの右端に出力列を追加**して保存する（列名は自由。複数列も可）
   - `run_id` 列と出力列だけを持つ**結果CSVを別に作る**
3. 感度指数を計算します。

   ```bat
   python run_sobol.py analyze --samples work\samples.csv --out-dir results\my_model
   python run_sobol.py analyze --samples work\samples.csv --results work\results.csv --out-dir results\my_model
   ```

> 行の並べ替えはしても構いません。ただし `run_id`・`block`・`sample_index` の列と入力の列は書き換えないでください。

### サンプル数 N の決め方

- モデルの計算回数は **N × (入力の数 + 2)** 回です。
- N は 16 以上の2のべき乗（512、1024、2048 …）にします。
- まず N = 1024 程度で計算し、画面の「収束の目安」で N/2 と N の差が大きければ、N を2倍にして計算し直します。

| 入力の数 | N = 1024 | N = 4096 |
|---|---|---|
| 3 | 5,120 回 | 20,480 回 |
| 5 | 7,168 回 | 28,672 回 |
| 10 | 12,288 回 | 49,152 回 |

### 主なオプション

| オプション | 対象 | 内容（既定値） |
|---|---|---|
| `--n` | check / run / sample | 基本サンプル数 N（check: 8192、その他: 1024） |
| `--sampler` | check / run / sample | `auto`: SciPy があれば Sobol列 / `sobol` / `random`（auto） |
| `--seed` | すべて | 乱数の種。同じ値なら同じ結果になる（12345） |
| `--bootstrap` | check / run / analyze | 信頼区間を求めるブートストラップの回数（1000） |
| `--confidence` | check / run / analyze | 信頼区間の水準（0.95） |
| `--out-dir` | run / analyze | 結果を保存するフォルダ（results） |
| `--outputs` | analyze | 計算する出力列を指定する（省略時はすべて） |
| `--function` | run | モデル関数の名前（model） |
| `--overwrite` | sample | 同名のサンプルCSVがあれば上書きする |
| `--tolerance` | check | 理論値との差の許容値（0.03） |

詳しくは `python run_sobol.py <コマンド> --help` で確認できます。

## 結果の見方

`--out-dir` に次のファイルを保存します（UTF-8 BOM付き。Excel でそのまま開けます）。

| ファイル | 内容 |
|---|---|
| `sobol_indices.csv` | 出力・入力ごとの `S1`、`S1_conf`、`ST`、`ST_conf`、`ST_minus_S1`。`_conf` は信頼区間の半幅（推定値 ± この値） |
| `convergence.csv` | 先頭 N/4・N/2・N 個のサンプルで計算し直した S1・ST（収束の確認用） |
| `samples_with_outputs.csv` | run のみ。計算に使った入力と出力 |

| 見るところ | 読み方 |
|---|---|
| **S1** | その入力だけで説明できる出力分散の割合 |
| **ST** | その入力が少しでも関わる出力分散の割合（交互作用を含む） |
| **ST − S1** が大きい | 他の入力との交互作用を通じて効いている |
| **ST ≈ 0** | ほぼ影響なし。固定値にできる候補（信頼区間の上限で判断する） |
| **S1 の合計 ≈ 1** | 足し算的なモデル。1 より小さいほど交互作用が大きい |
| 小さな負の値 | 推定誤差によるもの。0 とみなす |

画面の「お知らせ」は目安のしきい値で出しています。しきい値は `src/sobol_sensitivity/report.py` の定数で変更できます。

## テスト

```bat
python -m pip install pytest
python -m pytest
```

SALib がインストールされている環境では、推定値が SALib と一致するかも確認します（無い場合はそのテストだけスキップ）。

## ディレクトリ構成

```
sobol-sensitivity-analyzer/
├── run_sobol.py            # 起動スクリプト（インストール不要）
├── src/sobol_sensitivity/  # 本体（構成は design.md 参照）
├── examples/               # パラメータCSVとモデル関数の例
├── tests/                  # テスト
├── docs/                   # 学習用資料（PowerPoint）
├── requirements.md         # 要件
├── design.md               # 設計
├── issues.md               # 課題管理
├── requirements.txt        # pip でインストールするライブラリ
└── pyproject.toml          # Python バージョンとテストの設定
```

## 注意事項

- **入力どうしは独立**である前提です。相関のある入力には、このツールの結果をそのまま使わないでください。
- 結果は、入力に与えた**範囲と分布に左右されます**。設定の根拠を記録しておいてください。
- `--model` で指定した `.py` ファイルは、そのまま実行されます。**内容を確認した自分のファイルだけ**を指定してください。
- 社内データや計算結果をコミットしないでください。`results/` と `work/` は `.gitignore` で除外していますが、それ以外の場所に置いた場合は除外されません。

## 参考文献

- A. Saltelli et al. (2010). Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index. *Computer Physics Communications*, 181(2), 259–270.
- M. J. W. Jansen (1999). Analysis of variance designs for model output. *Computer Physics Communications*, 117(1–2), 35–43.
- T. Ishigami, T. Homma (1990). An importance quantification technique in uncertainty analysis for computer models. *Proc. ISUMA '90*, 398–403.
- J. Herman, W. Usher (2017). SALib: An open-source Python library for Sensitivity Analysis. *Journal of Open Source Software*, 2(9), 97.

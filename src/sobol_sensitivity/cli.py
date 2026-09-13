"""コマンドラインの入口

  check    Ishigami関数で、理論値と一致するか検算する
  run      Pythonで書いたモデル関数を使い、サンプル作成から指数の計算までまとめて行う
  sample   外部ツールで計算するための入力サンプルCSVを作る
  analyze  出力を追加したサンプルCSV（または結果CSV）から指数を計算する
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from .benchmark import ishigami, ishigami_analytic_indices, ishigami_parameters
from .errors import SobolInputError
from .estimators import BlockOutputs, compute_convergence, compute_sobol_indices
from .model_runner import DEFAULT_FUNCTION_NAME, DEFAULT_OUTPUT_NAME, evaluate_model, load_model_function
from .parameters import read_parameters_csv
from .report import (
    CONVERGENCE_FILE_NAME,
    INDICES_FILE_NAME,
    SAMPLES_WITH_OUTPUTS_FILE_NAME,
    format_summary,
    write_convergence_csv,
    write_indices_csv,
)
from .samples_file import read_block_outputs, write_samples_csv
from .sampling import SAMPLER_CHOICES, build_saltelli_design, split_stacked_outputs

MINIMUM_PYTHON_VERSION = (3, 14, 5)
DEFAULT_BASE_SAMPLES = 1024
DEFAULT_CHECK_BASE_SAMPLES = 8192
DEFAULT_BOOTSTRAP_COUNT = 1000
MIN_BOOTSTRAP_COUNT = 100
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_SEED = 12345
DEFAULT_OUTPUT_DIRECTORY = "results"
DEFAULT_CHECK_TOLERANCE = 0.03
EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_INPUT_ERROR = 2


def main(argv: list[str] | None = None) -> int:
    _make_console_output_safe()
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        required = ".".join(map(str, MINIMUM_PYTHON_VERSION))
        print(f"エラー: Python {required} 以上で実行してください（現在 {sys.version.split()[0]}）", file=sys.stderr)
        return EXIT_INPUT_ERROR
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except SobolInputError as error:
        print(f"エラー: {error}", file=sys.stderr)
        return EXIT_INPUT_ERROR


def _make_console_output_safe() -> None:
    # Windows の古いコンソールで表示できない文字があっても、処理を止めずに置き換えて表示する
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_sobol.py", description="Sobol感度解析（一次効果 S1・総合効果 ST）を計算します。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Ishigami関数で理論値と一致するか検算する")
    _add_sampling_options(check, DEFAULT_CHECK_BASE_SAMPLES)
    _add_analysis_options(check)
    check.add_argument("--tolerance", type=_positive_float, default=DEFAULT_CHECK_TOLERANCE,
                       help=f"理論値との差の許容値（既定: {DEFAULT_CHECK_TOLERANCE}）")
    check.set_defaults(handler=command_check)

    run = subparsers.add_parser("run", help="Pythonのモデル関数で、サンプル作成から指数計算までまとめて行う")
    run.add_argument("--params", required=True, help="入力パラメータ定義CSV")
    run.add_argument("--model", required=True, help="モデル関数を書いた .py ファイル")
    run.add_argument("--function", default=DEFAULT_FUNCTION_NAME, help=f"モデル関数の名前（既定: {DEFAULT_FUNCTION_NAME}）")
    _add_sampling_options(run, DEFAULT_BASE_SAMPLES)
    _add_analysis_options(run)
    _add_output_directory_option(run)
    run.set_defaults(handler=command_run)

    sample = subparsers.add_parser("sample", help="外部ツールで計算するための入力サンプルCSVを作る")
    sample.add_argument("--params", required=True, help="入力パラメータ定義CSV")
    sample.add_argument("--out", required=True, help="作成するサンプルCSVのパス")
    sample.add_argument("--overwrite", action="store_true", help="同名のファイルがあれば上書きする")
    _add_sampling_options(sample, DEFAULT_BASE_SAMPLES)
    sample.set_defaults(handler=command_sample)

    analyze = subparsers.add_parser("analyze", help="出力を追加したサンプルCSV（または結果CSV）から指数を計算する")
    analyze.add_argument("--samples", required=True, help="sample コマンドで作ったサンプルCSV")
    analyze.add_argument("--results", help="run_id 列と出力列を持つ結果CSV（出力をサンプルCSVに追加した場合は不要）")
    analyze.add_argument("--outputs", nargs="+", help="計算する出力列の名前（省略時はすべての出力列）")
    _add_analysis_options(analyze)
    _add_output_directory_option(analyze)
    analyze.set_defaults(handler=command_analyze)
    return parser


def _add_sampling_options(parser: argparse.ArgumentParser, default_base_samples: int) -> None:
    parser.add_argument("--n", type=int, default=default_base_samples,
                        help=f"基本サンプル数 N（2のべき乗）。モデル計算は N×(入力数+2) 回（既定: {default_base_samples}）")
    parser.add_argument("--sampler", choices=SAMPLER_CHOICES, default="auto",
                        help="auto: SciPy があれば Sobol列、無ければ通常の乱数（既定: auto）")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"乱数の種（既定: {DEFAULT_SEED}）")


def _add_analysis_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bootstrap", type=_bootstrap_count, default=DEFAULT_BOOTSTRAP_COUNT,
                        help=f"信頼区間を求めるブートストラップの回数（既定: {DEFAULT_BOOTSTRAP_COUNT}）")
    parser.add_argument("--confidence", type=_confidence_level, default=DEFAULT_CONFIDENCE_LEVEL,
                        help=f"信頼区間の水準（既定: {DEFAULT_CONFIDENCE_LEVEL}）")
    if not any(action.dest == "seed" for action in parser._actions):
        parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"乱数の種（既定: {DEFAULT_SEED}）")


def _add_output_directory_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIRECTORY,
                        help=f"結果を保存するフォルダ（既定: {DEFAULT_OUTPUT_DIRECTORY}）")


def _bootstrap_count(text: str) -> int:
    value = int(text)
    if value < MIN_BOOTSTRAP_COUNT:
        raise argparse.ArgumentTypeError(f"{MIN_BOOTSTRAP_COUNT} 以上を指定してください")
    return value


def _confidence_level(text: str) -> float:
    value = float(text)
    if not 0 < value < 1:
        raise argparse.ArgumentTypeError("0 より大きく 1 より小さい値を指定してください")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError("0 より大きい値を指定してください")
    return value


def command_check(args: argparse.Namespace) -> int:
    parameters = ishigami_parameters()
    design, notes = build_saltelli_design(parameters, args.n, args.sampler, args.seed)
    _print_notes(notes)
    values = evaluate_model(ishigami, design.parameter_names, design.stacked_inputs())[DEFAULT_OUTPUT_NAME]
    outputs = split_stacked_outputs(values, design.base_sample_count, len(parameters))
    indices = compute_sobol_indices(design.parameter_names, outputs, args.bootstrap, args.confidence, args.seed)
    expected_first, expected_total = ishigami_analytic_indices()

    print(f"Ishigami関数の検算（N = {design.base_sample_count}、サンプル: {design.sampler_used}、"
          f"モデル計算 {design.total_run_count} 回）")
    print("  入力    S1 推定   S1 理論   ST 推定   ST 理論")
    for index, name in enumerate(design.parameter_names):
        print(f"  {name:<6}{indices.first_order[index]:9.4f} {expected_first[index]:9.4f} "
              f"{indices.total_order[index]:9.4f} {expected_total[index]:9.4f}")
    max_difference = float(max(
        np.max(np.abs(indices.first_order - expected_first)),
        np.max(np.abs(indices.total_order - expected_total)),
    ))
    passed = max_difference <= args.tolerance
    print(f"判定: {'合格' if passed else '不合格'}（理論値との差 最大 {max_difference:.4f}、許容差 {args.tolerance}）")
    return EXIT_OK if passed else EXIT_CHECK_FAILED


def command_run(args: argparse.Namespace) -> int:
    parameters = read_parameters_csv(args.params)
    model_function = load_model_function(args.model, args.function)
    design, notes = build_saltelli_design(parameters, args.n, args.sampler, args.seed)
    _print_notes(notes)
    print(f"モデルを {design.total_run_count} 回計算します（N = {design.base_sample_count}、"
          f"入力 {len(parameters)} 個、サンプル: {design.sampler_used}）")
    stacked_outputs = evaluate_model(model_function, design.parameter_names, design.stacked_inputs())

    output_directory = Path(args.out_dir)
    samples_path = write_samples_csv(output_directory / SAMPLES_WITH_OUTPUTS_FILE_NAME, design, stacked_outputs)
    block_outputs = {
        name: split_stacked_outputs(values, design.base_sample_count, len(parameters))
        for name, values in stacked_outputs.items()
    }
    _analyze_and_save(design.parameter_names, block_outputs, args, output_directory)
    print(f"  入力と出力: {samples_path}")
    return EXIT_OK


def command_sample(args: argparse.Namespace) -> int:
    parameters = read_parameters_csv(args.params)
    output_path = Path(args.out)
    if output_path.exists() and not args.overwrite:
        raise SobolInputError(f"出力先がすでに存在します: {output_path}（上書きする場合は --overwrite を付けてください）")
    design, notes = build_saltelli_design(parameters, args.n, args.sampler, args.seed)
    _print_notes(notes)
    write_samples_csv(output_path, design)

    print(f"サンプルCSVを作成しました: {output_path}")
    print(f"  N = {design.base_sample_count}、入力 {len(parameters)} 個、"
          f"行数 {design.total_run_count} 行（= N × (入力数 + 2)）、サンプル: {design.sampler_used}")
    print("次の手順:")
    print("  1. 各行の入力値でモデルを計算し、右端に出力列（例: Y）を追加して保存する")
    print("     （または run_id 列と出力列だけの結果CSVを別に作る）")
    print(f"  2. python run_sobol.py analyze --samples {output_path}  （結果CSVを分けた場合は --results も指定）")
    return EXIT_OK


def command_analyze(args: argparse.Namespace) -> int:
    loaded = read_block_outputs(args.samples, args.results, args.outputs)
    print(f"読み込み: N = {loaded.base_sample_count}、入力 {len(loaded.parameter_names)} 個、"
          f"出力 {', '.join(loaded.outputs)}")
    _analyze_and_save(loaded.parameter_names, loaded.outputs, args, Path(args.out_dir))
    return EXIT_OK


def _analyze_and_save(
    parameter_names: tuple[str, ...], block_outputs: dict[str, BlockOutputs], args: argparse.Namespace,
    output_directory: Path,
) -> None:
    results = {}
    for output_name, outputs in block_outputs.items():
        try:
            indices = compute_sobol_indices(parameter_names, outputs, args.bootstrap, args.confidence, args.seed)
        except SobolInputError as error:
            raise SobolInputError(f"出力 '{output_name}': {error}") from error
        convergence = compute_convergence(outputs)
        results[output_name] = (indices, convergence)
        print(format_summary(output_name, indices, convergence))

    indices_path = write_indices_csv(output_directory / INDICES_FILE_NAME, results)
    convergence_path = write_convergence_csv(output_directory / CONVERGENCE_FILE_NAME, results)
    print("")
    print("保存しました:")
    print(f"  感度指数: {indices_path}")
    print(f"  収束確認: {convergence_path}")


def _print_notes(notes: list[str]) -> None:
    for note in notes:
        print(f"お知らせ: {note}")

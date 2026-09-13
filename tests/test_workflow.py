"""コマンド（check / run / sample / analyze）を通しで実行するテスト"""

import csv
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sobol_sensitivity import cli
from sobol_sensitivity.benchmark import ishigami, ishigami_analytic_indices

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIRECTORY = REPOSITORY_ROOT / "examples"
ISHIGAMI_PARAMETERS = EXAMPLES_DIRECTORY / "ishigami_parameters.csv"
BOOTSTRAP_ARGUMENTS = ["--bootstrap", "100"]
BASE_SAMPLES = "4096"
SMALL_BASE_SAMPLES = "16"
TOLERANCE = 0.03
INPUT_COLUMNS = slice(3, 6)  # run_id, block, sample_index の後ろにある x1〜x3
EXIT_INPUT_ERROR = 2


def _read_csv(path, encoding="utf-8-sig"):
    with Path(path).open(newline="", encoding=encoding) as file:
        return list(csv.reader(file))


def _write_csv(path, rows, encoding="utf-8-sig"):
    with Path(path).open("w", newline="", encoding=encoding) as file:
        csv.writer(file).writerows(rows)


def _read_indices(path):
    header, *body = _read_csv(path)
    records = [dict(zip(header, row)) for row in body]
    return {(record["output"], record["parameter"]): (float(record["S1"]), float(record["ST"])) for record in records}


def _assert_ishigami_indices(indices, output_name):
    expected_first, expected_total = ishigami_analytic_indices()
    for index, name in enumerate(("x1", "x2", "x3")):
        assert indices[(output_name, name)] == pytest.approx(
            (expected_first[index], expected_total[index]), abs=TOLERANCE
        )


def _ishigami_from_sample_rows(rows):
    inputs = np.array([[float(value) for value in row[INPUT_COLUMNS]] for row in rows])
    return ishigami({"x1": inputs[:, 0], "x2": inputs[:, 1], "x3": inputs[:, 2]})


def _create_samples(tmp_path, base_samples=BASE_SAMPLES):
    samples_path = tmp_path / "samples.csv"
    arguments = ["sample", "--params", str(ISHIGAMI_PARAMETERS), "--n", base_samples, "--out", str(samples_path)]
    assert cli.main(arguments) == 0
    return samples_path


def test_check_command_passes(capsys):
    assert cli.main(["check", "--n", BASE_SAMPLES, *BOOTSTRAP_ARGUMENTS]) == 0
    assert "合格" in capsys.readouterr().out


def test_launcher_script_runs_without_installation():
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "run_sobol.py"), "check", "--n", BASE_SAMPLES, *BOOTSTRAP_ARGUMENTS],
        capture_output=True, text=True, encoding="utf-8", cwd=REPOSITORY_ROOT, check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_run_command_with_example_model(tmp_path):
    exit_code = cli.main([
        "run", "--params", str(ISHIGAMI_PARAMETERS), "--model", str(EXAMPLES_DIRECTORY / "ishigami_model.py"),
        "--n", BASE_SAMPLES, *BOOTSTRAP_ARGUMENTS, "--out-dir", str(tmp_path),
    ])
    assert exit_code == 0
    _assert_ishigami_indices(_read_indices(tmp_path / "sobol_indices.csv"), "Y")
    assert (tmp_path / "convergence.csv").is_file()
    assert len(_read_csv(tmp_path / "samples_with_outputs.csv")) == int(BASE_SAMPLES) * 5 + 1


def test_run_command_with_multiple_outputs(tmp_path):
    exit_code = cli.main([
        "run", "--params", str(EXAMPLES_DIRECTORY / "multi_output_parameters.csv"),
        "--model", str(EXAMPLES_DIRECTORY / "multi_output_model.py"),
        "--n", BASE_SAMPLES, *BOOTSTRAP_ARGUMENTS, "--out-dir", str(tmp_path),
    ])
    assert exit_code == 0
    indices = _read_indices(tmp_path / "sobol_indices.csv")

    # linear_output: 交互作用なし。指数は各項の分散の比（1/12 : 4 : 1/6）
    variances = {"uniform_input": 1 / 12, "normal_input": 4.0, "triangular_input": 1 / 6}
    total_variance = sum(variances.values())
    for name, variance in variances.items():
        ratio = variance / total_variance
        assert indices[("linear_output", name)] == pytest.approx((ratio, ratio), abs=TOLERANCE)

    # interaction_output = u × t: S_t = 0.75、ST_u = 0.25、ST_t = 1、normal_input は無関係
    assert indices[("interaction_output", "uniform_input")] == pytest.approx((0.0, 0.25), abs=TOLERANCE)
    assert indices[("interaction_output", "normal_input")] == pytest.approx((0.0, 0.0), abs=TOLERANCE)
    assert indices[("interaction_output", "triangular_input")] == pytest.approx((0.75, 1.0), abs=TOLERANCE)


def test_sample_then_analyze_with_output_column_added(tmp_path):
    samples_path = _create_samples(tmp_path)
    header, *body = _read_csv(samples_path)
    assert header == ["run_id", "block", "sample_index", "x1", "x2", "x3"]

    # 外部ツールで計算し、サンプルCSVの右端に出力列を追加した状況を再現する
    outputs = _ishigami_from_sample_rows(body)
    _write_csv(samples_path, [header + ["Y"], *[row + [repr(float(value))] for row, value in zip(body, outputs)]])

    out_directory = tmp_path / "results"
    arguments = ["analyze", "--samples", str(samples_path), *BOOTSTRAP_ARGUMENTS, "--out-dir", str(out_directory)]
    assert cli.main(arguments) == 0
    _assert_ishigami_indices(_read_indices(out_directory / "sobol_indices.csv"), "Y")


def test_analyze_with_separate_shift_jis_results_in_shuffled_order(tmp_path):
    samples_path = _create_samples(tmp_path)
    body = _read_csv(samples_path)[1:]
    outputs = _ishigami_from_sample_rows(body)
    result_rows = [[row[0], repr(float(value))] for row, value in zip(body, outputs)]
    random.Random(0).shuffle(result_rows)
    results_path = tmp_path / "results.csv"
    _write_csv(results_path, [["run_id", "出力Y"], *result_rows], encoding="cp932")

    out_directory = tmp_path / "out"
    arguments = [
        "analyze", "--samples", str(samples_path), "--results", str(results_path),
        *BOOTSTRAP_ARGUMENTS, "--out-dir", str(out_directory),
    ]
    assert cli.main(arguments) == 0
    _assert_ishigami_indices(_read_indices(out_directory / "sobol_indices.csv"), "出力Y")


def test_analyze_reports_missing_outputs(tmp_path, capsys):
    samples_path = _create_samples(tmp_path, SMALL_BASE_SAMPLES)
    body = _read_csv(samples_path)[1:]
    results_path = tmp_path / "results.csv"
    _write_csv(results_path, [["run_id", "Y"], *[[row[0], "1.0"] for row in body[:-3]]])
    capsys.readouterr()

    arguments = ["analyze", "--samples", str(samples_path), "--results", str(results_path), "--out-dir", str(tmp_path / "out")]
    assert cli.main(arguments) == EXIT_INPUT_ERROR
    assert "3 行分ありません" in capsys.readouterr().err


def test_sample_does_not_overwrite_without_flag(tmp_path):
    samples_path = tmp_path / "samples.csv"
    samples_path.write_text("大事な結果", encoding="utf-8")
    arguments = ["sample", "--params", str(ISHIGAMI_PARAMETERS), "--n", SMALL_BASE_SAMPLES, "--out", str(samples_path)]
    assert cli.main(arguments) == EXIT_INPUT_ERROR
    assert samples_path.read_text(encoding="utf-8") == "大事な結果"
    assert cli.main([*arguments, "--overwrite"]) == 0


def test_run_reports_wrong_output_shape(tmp_path, capsys):
    model_path = tmp_path / "broken_model.py"
    model_path.write_text("def model(inputs):\n    return inputs['x1'][:10]\n", encoding="utf-8")
    arguments = [
        "run", "--params", str(ISHIGAMI_PARAMETERS), "--model", str(model_path),
        "--n", SMALL_BASE_SAMPLES, "--out-dir", str(tmp_path / "out"),
    ]
    assert cli.main(arguments) == EXIT_INPUT_ERROR
    assert "1次元配列を返してください" in capsys.readouterr().err

"""計算結果のCSV保存と、画面に表示する要約の作成"""

from pathlib import Path

import numpy as np

from .csv_io import write_csv
from .estimators import ConvergenceStep, SobolIndices

INDICES_FILE_NAME = "sobol_indices.csv"
CONVERGENCE_FILE_NAME = "convergence.csv"
SAMPLES_WITH_OUTPUTS_FILE_NAME = "samples_with_outputs.csv"
CSV_DECIMALS = 6
# 以下のしきい値は画面のお知らせを出すための目安。目的に応じて見直す
CONVERGENCE_WARNING_DIFFERENCE = 0.05  # N/2 と N の推定値の差がこれを超えたら N を増やすよう促す
FIXING_CANDIDATE_UPPER_LIMIT = 0.01  # 総合効果の信頼区間の上限がこれ未満なら「固定できる候補」
INTERACTION_NOTE_DIFFERENCE = 0.05  # ST − S1 がこれを超え、信頼区間でも区別できれば「交互作用あり」

AnalysisResults = dict[str, tuple[SobolIndices, list[ConvergenceStep]]]


def write_indices_csv(path: str | Path, results: AnalysisResults) -> Path:
    header = ["output", "parameter", "S1", "S1_conf", "ST", "ST_conf", "ST_minus_S1", "base_samples", "confidence_level"]
    rows = []
    for output_name, (indices, _) in results.items():
        for index, parameter_name in enumerate(indices.parameter_names):
            first, total = indices.first_order[index], indices.total_order[index]
            rows.append([
                output_name,
                parameter_name,
                _format_number(first),
                _format_number(indices.first_order_conf[index]),
                _format_number(total),
                _format_number(indices.total_order_conf[index]),
                _format_number(total - first),
                str(indices.base_sample_count),
                str(indices.confidence_level),
            ])
    return write_csv(path, header, rows)


def write_convergence_csv(path: str | Path, results: AnalysisResults) -> Path:
    header = ["output", "base_samples", "parameter", "S1", "ST"]
    rows = []
    for output_name, (indices, convergence) in results.items():
        for step in convergence:
            for index, parameter_name in enumerate(indices.parameter_names):
                rows.append([
                    output_name,
                    str(step.base_sample_count),
                    parameter_name,
                    _format_number(step.first_order[index]),
                    _format_number(step.total_order[index]),
                ])
    return write_csv(path, header, rows)


def _format_number(value: float) -> str:
    return f"{value:.{CSV_DECIMALS}f}"


def format_summary(output_name: str, indices: SobolIndices, convergence: list[ConvergenceStep]) -> str:
    names = indices.parameter_names
    name_width = max(len(name) for name in names) + 2
    lines = [
        "",
        f"=== 出力: {output_name}（N = {indices.base_sample_count}、信頼区間 {indices.confidence_level:.0%}）===",
        f"  {'入力':<{name_width - 2}}  S1 一次効果          ST 総合効果          ST-S1",
    ]
    order = np.argsort(-indices.total_order)  # 総合効果の大きい順に並べる
    for index in order:
        first, total = indices.first_order[index], indices.total_order[index]
        lines.append(
            f"  {names[index]:<{name_width}}"
            f"{first:7.4f} +/- {indices.first_order_conf[index]:6.4f}   "
            f"{total:7.4f} +/- {indices.total_order_conf[index]:6.4f}   "
            f"{total - first:7.4f}"
        )
    first_sum = float(np.sum(indices.first_order))
    lines.append(f"  S1 の合計 = {first_sum:.3f}（1 に近いほど足し算的なモデル。1 との差は交互作用の目安）")

    notes = [*_parameter_notes(indices), *_convergence_notes(convergence, lines)]
    if notes:
        lines.append("  お知らせ:")
        lines.extend(f"   - {note}" for note in notes)
    return "\n".join(lines)


def _parameter_notes(indices: SobolIndices) -> list[str]:
    notes = []
    for index, name in enumerate(indices.parameter_names):
        first, total = indices.first_order[index], indices.total_order[index]
        first_conf, total_conf = indices.first_order_conf[index], indices.total_order_conf[index]
        if total + total_conf < FIXING_CANDIDATE_UPPER_LIMIT:
            notes.append(f"{name}: 総合効果がほぼ 0 です（上限 {total + total_conf:.3f}）。固定値にできる候補です")
        elif total - first > max(INTERACTION_NOTE_DIFFERENCE, first_conf + total_conf):
            notes.append(f"{name}: ST と S1 の差が {total - first:.3f} あり、他の入力との交互作用を通じて効いています")
        if first - total > first_conf + total_conf:
            notes.append(f"{name}: S1 が ST を上回っています（推定誤差）。N を増やして再計算してください")
        if first + first_conf < 0 or total + total_conf < 0:
            notes.append(f"{name}: 信頼区間全体が負になっています（推定誤差）。N を増やして再計算してください")
    return notes


def _convergence_notes(convergence: list[ConvergenceStep], lines: list[str]) -> list[str]:
    if len(convergence) < 2:
        return []
    previous, latest = convergence[-2], convergence[-1]
    difference = float(
        max(
            np.max(np.abs(latest.first_order - previous.first_order)),
            np.max(np.abs(latest.total_order - previous.total_order)),
        )
    )
    lines.append(
        f"  収束の目安: N = {previous.base_sample_count} と N = {latest.base_sample_count} の差は最大 {difference:.3f}"
    )
    if difference > CONVERGENCE_WARNING_DIFFERENCE:
        return ["N/2 と N の結果の差が大きいため、N を2倍にして再計算し、値が落ち着くか確認してください"]
    return []

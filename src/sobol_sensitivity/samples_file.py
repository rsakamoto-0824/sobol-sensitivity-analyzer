"""サンプルCSVの書き出しと、出力値の読み込み

サンプルCSVの形式（1行 = モデル計算1回分）:
    run_id, block, sample_index, 入力1, …, 入力d [, 出力1, …]
block は A, B, AB_1 … AB_d。AB_i は行列 A の i 番目の入力だけを B の値に入れ替えたもの。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .csv_io import META_COLUMNS, CsvTable, parse_float, parse_int, read_csv_table, write_csv
from .errors import SobolInputError
from .estimators import BlockOutputs
from .parameters import MIN_PARAMETER_COUNT
from .sampling import (
    AB_BLOCK_PREFIX,
    BLOCK_A,
    BLOCK_B,
    SaltelliDesign,
    ab_block_name,
    split_stacked_outputs,
    validate_base_sample_count,
)

RUN_ID_COLUMN, BLOCK_COLUMN, SAMPLE_INDEX_COLUMN = META_COLUMNS
MAX_LISTED_RUN_IDS = 5


@dataclass(frozen=True)
class LoadedOutputs:
    parameter_names: tuple[str, ...]
    base_sample_count: int
    outputs: dict[str, BlockOutputs]


@dataclass(frozen=True)
class _SampleMeta:
    line_number: int
    run_id: int
    block: str
    sample_index: int


def write_samples_csv(path: str | Path, design: SaltelliDesign, outputs: dict[str, np.ndarray] | None = None) -> Path:
    output_names = list(outputs or {})
    header = [*META_COLUMNS, *design.parameter_names, *output_names]
    block_names = design.block_names()
    base_count = design.base_sample_count
    rows = []
    for position, input_values in enumerate(design.stacked_inputs()):
        row = [str(position + 1), block_names[position // base_count], str(position % base_count + 1)]
        row += [repr(float(value)) for value in input_values]
        row += [repr(float(outputs[name][position])) for name in output_names]
        rows.append(row)
    return write_csv(path, header, rows)


def read_block_outputs(
    samples_path: str | Path, results_path: str | Path | None = None, output_names: list[str] | None = None
) -> LoadedOutputs:
    """サンプルCSV（右端に出力列を追加したもの）または、サンプルCSV＋結果CSV（run_id と出力列）から出力を読む"""
    samples = read_csv_table(samples_path)
    if tuple(samples.header[: len(META_COLUMNS)]) != META_COLUMNS:
        raise SobolInputError(
            f"{samples.path.name}: 先頭の列は {', '.join(META_COLUMNS)} の順にしてください（sample コマンドで作ったCSVを使ってください）"
        )
    if not samples.rows:
        raise SobolInputError(f"{samples.path.name}: データ行がありません")

    metas = [_parse_meta(samples.path.name, line_number, row) for line_number, row in samples.rows]
    _rows_by_run_id(samples)  # run_id の重複チェック
    dimension, base_count = _check_block_structure(metas, samples.path.name)
    input_end = len(META_COLUMNS) + dimension
    if len(samples.header) < input_end:
        raise SobolInputError(f"{samples.path.name}: 入力列が {dimension} 列必要ですが、足りません")
    parameter_names = tuple(samples.header[len(META_COLUMNS) : input_end])

    if results_path is None:
        value_table = samples
        candidate_outputs = samples.header[input_end:]
    else:
        value_table = read_csv_table(results_path)
        if RUN_ID_COLUMN not in value_table.header:
            raise SobolInputError(f"{value_table.path.name}: 結果CSVには {RUN_ID_COLUMN} 列が必要です")
        excluded_columns = {*META_COLUMNS, *parameter_names}
        candidate_outputs = [column for column in value_table.header if column not in excluded_columns]
    selected_outputs = _select_output_columns(candidate_outputs, output_names, value_table.path.name)

    rows_by_run_id = _rows_by_run_id(value_table)
    block_order = {name: index for index, name in enumerate([BLOCK_A, BLOCK_B, *map(ab_block_name, range(dimension))])}
    outputs = {}
    for output_name in selected_outputs:
        stacked_values = np.empty(base_count * (dimension + 2))
        missing_run_ids = []
        for meta in metas:
            found = rows_by_run_id.get(meta.run_id)
            text = found[1].get(output_name, "") if found else ""
            if not text:
                missing_run_ids.append(meta.run_id)
                continue
            position = block_order[meta.block] * base_count + meta.sample_index - 1
            stacked_values[position] = parse_float(text, f"{value_table.path.name} {found[0]}行目の {output_name} 列")
        if missing_run_ids:
            listed = ", ".join(map(str, missing_run_ids[:MAX_LISTED_RUN_IDS]))
            more = " ほか" if len(missing_run_ids) > MAX_LISTED_RUN_IDS else ""
            raise SobolInputError(
                f"出力 '{output_name}' の値が {len(missing_run_ids)} 行分ありません（run_id: {listed}{more}）"
            )
        outputs[output_name] = split_stacked_outputs(stacked_values, base_count, dimension)
    return LoadedOutputs(parameter_names, base_count, outputs)


def _parse_meta(file_name: str, line_number: int, row: dict[str, str]) -> _SampleMeta:
    location = f"{file_name} {line_number}行目"
    return _SampleMeta(
        line_number=line_number,
        run_id=parse_int(row[RUN_ID_COLUMN], f"{location} の {RUN_ID_COLUMN}"),
        block=row[BLOCK_COLUMN],
        sample_index=parse_int(row[SAMPLE_INDEX_COLUMN], f"{location} の {SAMPLE_INDEX_COLUMN}"),
    )


def _check_block_structure(metas: list[_SampleMeta], file_name: str) -> tuple[int, int]:
    """block 列と sample_index 列が A, B, AB_1…AB_d × 1…N の形になっているか確かめ、(d, N) を返す"""
    indices_by_block: dict[str, list[int]] = {}
    for meta in metas:
        indices_by_block.setdefault(meta.block, []).append(meta.sample_index)
    dimension = sum(1 for name in indices_by_block if name.startswith(AB_BLOCK_PREFIX))
    expected_blocks = [BLOCK_A, BLOCK_B, *map(ab_block_name, range(dimension))]
    if set(indices_by_block) != set(expected_blocks):
        raise SobolInputError(
            f"{file_name}: block 列の値が想定と違います（想定: {', '.join(expected_blocks)} / 実際: {', '.join(sorted(indices_by_block))}）"
        )
    if dimension < MIN_PARAMETER_COUNT:
        raise SobolInputError(f"{file_name}: 入力が {MIN_PARAMETER_COUNT} 個以上必要です（AB_ ブロックが {dimension} 個）")

    base_count = len(indices_by_block[BLOCK_A])
    for block_name in expected_blocks:
        if sorted(indices_by_block[block_name]) != list(range(1, base_count + 1)):
            raise SobolInputError(
                f"{file_name}: block {block_name} の sample_index が 1〜{base_count} の連番になっていません（行の抜けや重複がないか確認してください）"
            )
    validate_base_sample_count(base_count)
    return dimension, base_count


def _rows_by_run_id(table: CsvTable) -> dict[int, tuple[int, dict[str, str]]]:
    rows: dict[int, tuple[int, dict[str, str]]] = {}
    for line_number, row in table.rows:
        run_id = parse_int(row[RUN_ID_COLUMN], f"{table.path.name} {line_number}行目の {RUN_ID_COLUMN}")
        if run_id in rows:
            raise SobolInputError(f"{table.path.name}: {RUN_ID_COLUMN} {run_id} が重複しています")
        rows[run_id] = (line_number, row)
    return rows


def _select_output_columns(candidates: list[str], requested: list[str] | None, file_name: str) -> list[str]:
    if requested:
        unknown = [name for name in requested if name not in candidates]
        if unknown:
            raise SobolInputError(
                f"{file_name}: 指定した出力列が見つかりません: {', '.join(unknown)}（候補: {', '.join(candidates) or 'なし'}）"
            )
        return list(requested)
    if not candidates:
        raise SobolInputError(
            f"{file_name}: 出力列がありません。サンプルCSVの右端に出力列を追加するか、--results で結果CSVを指定してください"
        )
    return list(candidates)

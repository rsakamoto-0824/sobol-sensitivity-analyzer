"""CSVの読み書き

Excelで編集することを想定し、UTF-8（BOM付き・なし）と Shift_JIS（cp932）の両方を読めるようにする。
書き出しは、Excelで開いても文字化けしない UTF-8（BOM付き）にする。
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from .errors import SobolInputError

READ_ENCODINGS = ("utf-8-sig", "cp932")
WRITE_ENCODING = "utf-8-sig"
# サンプルCSVの先頭に置く管理用の列。入力名・出力名には使えない
META_COLUMNS = ("run_id", "block", "sample_index")


@dataclass(frozen=True)
class CsvTable:
    path: Path
    header: list[str]
    rows: list[tuple[int, dict[str, str]]]  # (ファイル上の行番号, 列名 → 前後の空白を除いたセルの文字列)


def read_csv_table(path: str | Path) -> CsvTable:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise SobolInputError(f"ファイルが見つかりません: {csv_path}")

    raw_rows = _read_raw_rows(csv_path)
    if not raw_rows or not any(cell.strip() for cell in raw_rows[0][1]):
        raise SobolInputError(f"{csv_path.name}: 1行目に列名がありません")
    header = _clean_header(raw_rows[0][1], csv_path.name)

    rows = []
    for line_number, cells in raw_rows[1:]:
        if not any(cell.strip() for cell in cells):
            continue
        if any(cell.strip() for cell in cells[len(header):]):
            raise SobolInputError(f"{csv_path.name} {line_number}行目: 列名のない列に値があります")
        padded_cells = cells + [""] * (len(header) - len(cells))
        rows.append((line_number, {name: cell.strip() for name, cell in zip(header, padded_cells)}))
    return CsvTable(csv_path, header, rows)


def _read_raw_rows(csv_path: Path) -> list[tuple[int, list[str]]]:
    for encoding in READ_ENCODINGS:
        try:
            with csv_path.open(newline="", encoding=encoding) as file:
                reader = csv.reader(file)
                return [(reader.line_num, cells) for cells in reader]
        except UnicodeDecodeError:
            continue
    raise SobolInputError(f"{csv_path.name}: 文字コードを判別できません（UTF-8 か Shift_JIS で保存してください）")


def _clean_header(cells: list[str], file_name: str) -> list[str]:
    header = [cell.strip() for cell in cells]
    # Excelで保存すると末尾に空の列が付くことがあるので取り除く
    while header and not header[-1]:
        header.pop()
    if "" in header:
        raise SobolInputError(f"{file_name}: 列名が空の列があります（1行目を確認してください）")
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise SobolInputError(f"{file_name}: 列名が重複しています: {', '.join(duplicates)}")
    return header


def write_csv(path: str | Path, header: list[str], rows: list[list[str]]) -> Path:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with csv_path.open("w", newline="", encoding=WRITE_ENCODING) as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)
    except PermissionError as error:
        raise SobolInputError(
            f"ファイルに書き込めません（Excelで開いたままになっていないか確認してください）: {csv_path}"
        ) from error
    return csv_path


def parse_float(text: str, description: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise SobolInputError(f"{description} を数値として読めません: '{text}'") from None
    if not math.isfinite(value):
        raise SobolInputError(f"{description} が有限の数値ではありません: '{text}'")
    return value


def parse_int(text: str, description: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise SobolInputError(f"{description} を整数として読めません: '{text}'") from None

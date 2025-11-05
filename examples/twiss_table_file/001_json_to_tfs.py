from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional


SCALAR_LABEL_MAP_DEFAULT: Dict[str, str] = {
    "qx": "Q1",
    "qy": "Q2",
    "dqx": "DQ1",
    "dqy": "DQ2",
}

FLOAT_COLUMN_FORMAT_DEFAULT: Dict[str, Dict[str, int]] = {
    "s": {"width": 15, "precision": 6},
    "betx": {"width": 18, "precision": 8},
    "bety": {"width": 18, "precision": 8},
}

DEFAULT_FLOAT_WIDTH = 18
DEFAULT_FLOAT_PRECISION = 10


@dataclass
class ColumnSpec:
    header: str
    type_code: str
    width: int
    formatter: Callable[[int], str]


def _infer_column_order(columns: Dict[str, Iterable]) -> List[str]:
    ordered: List[str] = []
    if "name" in columns:
        ordered.append("name")
    for key in columns.keys():  # preserves insertion order
        if key != "name":
            ordered.append(key)
    return ordered


def _make_float_formatter(values: List[Optional[float]], width: int, precision: int) -> Callable[[int], str]:
    def format_value(idx: int) -> str:
        value = values[idx]
        if value is None:
            return "".rjust(width)
        return f"{value:>{width}.{precision}f}"

    return format_value


def _make_generic_float_formatter(values: List[Optional[float]], width: int) -> Callable[[int], str]:
    def format_value(idx: int) -> str:
        value = values[idx]
        if value is None:
            return "".rjust(width)
        return f"{value:>{width}.10g}"

    return format_value


def _make_string_formatter(values: List[Optional[str]], width: int, transform: Callable[[str], str]) -> Callable[[int], str]:
    def format_value(idx: int) -> str:
        value = values[idx]
        if value is None:
            return "".rjust(width)
        return f"{transform(value):<{width}}"

    return format_value


def _format_scalar_line(label: str, value: Optional[float]) -> str:
    value_str = "".rjust(20) if value is None else f"{value:>20.12g}"
    return f"@ {label:<16} %le {value_str}"


def _build_column_specs(
    columns: Dict[str, List],
    *,
    float_column_format: Dict[str, Dict[str, int]],
    default_float_width: int,
    default_float_precision: int,
) -> List[ColumnSpec]:
    specs: List[ColumnSpec] = []
    names: List[str] = [str(value) for value in columns.get("name", [])]

    if names:
        name_width = max([20] + [len(f'"{value.upper()}"') for value in names])
        specs.append(
            ColumnSpec(
                header="NAME",
                type_code="%s",
                width=name_width,
                formatter=_make_string_formatter(names, name_width, lambda v: f'"{v.upper()}"'),
            )
        )

    for column_name in _infer_column_order(columns):
        if column_name == "name":
            continue

        values = columns[column_name]
        normalized = column_name.lower()

        if all((v is None or isinstance(v, (int, float))) for v in values):
            config = float_column_format.get(normalized)
            width = config["width"] if config else default_float_width
            precision = config["precision"] if config else default_float_precision
            formatter = _make_float_formatter(values, width, precision)
            specs.append(
                ColumnSpec(
                    header=normalized.upper(),
                    type_code="%le",
                    width=width,
                    formatter=formatter,
                )
            )
        else:
            str_values = ["" if v is None else str(v) for v in values]
            text_width = max([20] + [len(f'"{value}"') for value in str_values]) if str_values else 20
            specs.append(
                ColumnSpec(
                    header=normalized.upper(),
                    type_code="%s",
                    width=text_width,
                    formatter=_make_string_formatter(str_values, text_width, lambda v: f'"{v}"'),
                )
            )

    return specs


def _build_table_header(specs: List[ColumnSpec]) -> List[str]:
    header_cells = [
        f"{spec.header:<{spec.width}}" if spec.type_code == "%s" else f"{spec.header:>{spec.width}}"
        for spec in specs
    ]
    type_cells = [
        f"{spec.type_code:<{spec.width}}" if spec.type_code == "%s" else f"{spec.type_code:>{spec.width}}"
        for spec in specs
    ]
    header_line = "* " + " ".join(header_cells)
    type_line = "$ " + " ".join(type_cells)
    return [header_line.rstrip(), type_line.rstrip()]


def _build_table_rows(specs: List[ColumnSpec], row_count: int) -> List[str]:
    rows: List[str] = []
    for idx in range(row_count):
        cells = [spec.formatter(idx) for spec in specs]
        rows.append(" " + " ".join(cells).rstrip())
    return rows


def _build_scalar_section(
    scalar: Dict[str, float], *, scalar_label_map: Dict[str, str]
) -> List[str]:
    lines: List[str] = []
    for key, value in scalar.items():
        label = scalar_label_map.get(key.lower(), key.upper())
        lines.append(_format_scalar_line(label, value))
    return lines


def _validate_column_lengths(columns: Dict[str, List]) -> int:
    column_lengths = {key: len(values) for key, values in columns.items()}
    if not column_lengths:
        return 0

    unique_lengths = set(column_lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"Inconsistent column sizes detected: {column_lengths}")
    return unique_lengths.pop()


def prepare_tfs_lines(
    scalar: Dict[str, float],
    columns: Dict[str, List],
    *,
    scalar_label_map: Optional[Dict[str, str]] = None,
    float_column_format: Optional[Dict[str, Dict[str, int]]] = None,
    default_float_width: int = DEFAULT_FLOAT_WIDTH,
    default_float_precision: int = DEFAULT_FLOAT_PRECISION,
) -> tuple[List[str], List[str], List[ColumnSpec]]:
    row_count = _validate_column_lengths(columns)
    scalar_label_map = scalar_label_map or SCALAR_LABEL_MAP_DEFAULT
    float_column_format = float_column_format or FLOAT_COLUMN_FORMAT_DEFAULT

    specs = _build_column_specs(
        columns,
        float_column_format=float_column_format,
        default_float_width=default_float_width,
        default_float_precision=default_float_precision,
    )
    headers = _build_scalar_section(scalar, scalar_label_map=scalar_label_map)

    lines: List[str] = []
    lines.extend(headers)
    if specs:
        lines.extend(_build_table_header(specs))
        lines.extend(_build_table_rows(specs, row_count))

    return lines, headers, specs


def write_tfs_from_json(json_path: Path, output_path: Path) -> None:
    data = json.loads(json_path.read_text())
    scalar = data.get("scalar", {})
    columns = data.get("col", {})

    if not isinstance(columns, dict) or not columns:
        raise ValueError("JSON file does not contain column data under 'col'.")

    lines, _, _ = prepare_tfs_lines(scalar, columns)

    output_path.write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    input_json = Path(__file__).parent / "twiss_lhcb1_xtrack.json"
    output_tfs = Path(__file__).parent / "twiss_lhcb1_xtrack.tfs"

    write_tfs_from_json(input_json, output_tfs)

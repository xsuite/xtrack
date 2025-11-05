from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _parse_header_line(line: str) -> Tuple[str, str, str]:
    parts = shlex.split(line)
    if len(parts) < 4 or parts[0] != '@':
        raise ValueError(f"Invalid header line: {line}")
    label = parts[1].lower()
    type_code = parts[2]
    value_str = parts[3]
    if type_code.lower().startswith('%l'):
        value = float(value_str)
    elif type_code.lower().startswith('%i'):
        value = int(value_str)
    else:
        value = value_str.strip('"')
    return label, type_code, value


def _split_columns_line(line: str) -> List[str]:
    return shlex.split(line)


def read_tfs(path: Path) -> Tuple[Dict[str, float], Dict[str, List]]:
    scalar: Dict[str, float] = {}
    column_headers: List[str] = []
    column_types: List[str] = []
    data_start_line: int | None = None

    with path.open() as f:
        for idx, raw_line in enumerate(f):
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith('@'):
                label, _, value = _parse_header_line(stripped)
                scalar[label] = value
            elif stripped.startswith('*'):
                column_headers = [name.lower() for name in _split_columns_line(stripped[1:].strip())]
            elif stripped.startswith('$'):
                column_types = _split_columns_line(stripped[1:].strip())
                if len(column_types) != len(column_headers):
                    raise ValueError("Column type count does not match headers")
                data_start_line = idx + 1
                break

    if not column_headers or data_start_line is None:
        raise ValueError("Failed to locate table headers in TFS file")

    df = pd.read_csv(
        path,
        sep='\s+',
        skiprows=data_start_line,
        names=column_headers,
        engine='python',
    )
    df = df.dropna(how='all')

    columns: Dict[str, List] = {}
    for name, type_code in zip(column_headers, column_types):
        tcode = type_code.lower()
        if tcode.startswith('%s'):
            series = df[name]
            processed: List[str | None] = []
            for value in series:
                if pd.isna(value):
                    processed.append(None)
                else:
                    processed.append(str(value).strip('"'))
            columns[name] = processed
        else:
            numeric_series = pd.to_numeric(df[name], errors='coerce')
            columns[name] = [value if pd.notna(value) else None for value in numeric_series.tolist()]

    return scalar, columns


scalar, columns = read_tfs(Path("./twiss_lhcb1_xtrack.tfs"))
print("Scalars:\n", scalar)
print("\nColumns keys:\n", list(columns.keys()))

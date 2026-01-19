#!/usr/bin/env python3
"""
parquet_to_json.py

Parquet -> pretty JSON object indexed by row number:
{
  "0": {...},
  "1": {...}
}
"""

import argparse
import json
import math
import os
from datetime import date, datetime
from decimal import Decimal

import pandas as pd


def json_safe(obj):
    """Recursively convert objects into JSON-serializable Python types."""
    if obj is None:
        return None

    # Handle pandas missing (NaN/NaT/etc.)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    # numpy handling
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return [json_safe(x) for x in obj.tolist()]
        if isinstance(obj, np.generic):
            return json_safe(obj.item())
    except Exception:
        pass

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, Decimal):
        return str(obj)

    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [json_safe(x) for x in obj]

    return obj


def parquet_to_indexed_json(parquet_path: str, json_path: str, indent: int = 2):
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Input parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    df = df.where(pd.notnull(df), None)

    records = df.to_dict(orient="records")

    # Build {"0": {...}, "1": {...}} (JSON keys must be strings)
    indexed = {str(i): {k: json_safe(v) for k, v in row.items()} for i, row in enumerate(records)}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(indexed, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser(description="Convert Parquet to indexed pretty JSON.")
    ap.add_argument("parquet", help="Input .parquet file path")
    ap.add_argument("-o", "--out", help="Output .json path (default: input name with .json)")
    ap.add_argument("--indent", type=int, default=2, help="Pretty-print indent level (default: 2)")
    args = ap.parse_args()

    parquet_path = args.parquet
    json_path = args.out or (os.path.splitext(parquet_path)[0] + ".json")

    parquet_to_indexed_json(parquet_path, json_path, indent=args.indent)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import pandas as pd


def add_synergy_columns(df):
    if "coalition" not in df.columns or "objective" not in df.columns:
        return df

    df = df.copy()
    df["coalition"] = df["coalition"].astype(str)
    df["objective"] = pd.to_numeric(df["objective"], errors="coerce")

    standalone_costs = (
        df[df["coalition"].str.len() == 1]
        .dropna(subset=["objective"])
        .drop_duplicates(subset=["coalition"], keep="first")
        .set_index("coalition")["objective"]
        .to_dict()
    )

    standalone_values = []
    synergy_values = []

    for row in df.to_dict("records"):
        coalition = str(row["coalition"])
        objective = row.get("objective")

        if pd.isna(objective) or not all(member in standalone_costs for member in coalition):
            standalone_values.append(None)
            synergy_values.append(None)
            continue

        standalone = sum(standalone_costs[member] for member in coalition)
        cost_savings = standalone - float(objective)
        synergy = cost_savings / float(objective) if float(objective) else None

        standalone_values.append(standalone)
        synergy_values.append(synergy)

    df["standalone_cost"] = standalone_values
    df["synergy"] = synergy_values

    return df


def merge_csvs(input_dir, pattern, output):
    paths = sorted(Path(input_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {Path(input_dir) / pattern}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError("Matched files were empty.")

    merged = pd.concat(frames, ignore_index=True)
    merged = add_synergy_columns(merged)

    sort_cols = [c for c in ["coalition_size", "coalition", "wind_farm"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"Wrote {len(merged)} rows to {output}")


def build_parser():
    parser = argparse.ArgumentParser(description="Merge case-study node CSV files.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    merge_csvs(args.input_dir, args.pattern, args.output)


if __name__ == "__main__":
    main()

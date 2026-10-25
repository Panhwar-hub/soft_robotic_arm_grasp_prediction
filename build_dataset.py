"""Rebuild the feature-table CSV from all raw recordings under the three
class folders. Run this whenever new CSVs are added."""
import glob
import os

import pandas as pd

from feature_extraction import NON_MODEL_COLUMNS, extract_features_from_df

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Soft_Robotic_Arm/
CLASSES = ["Ball", "Square_Box", "Triangular"]

def main():
    """Build recording-level feature tables from the local class folders."""
    rows = []
    skipped = []
    for cls in CLASSES:
        for fp in sorted(glob.glob(os.path.join(ROOT, cls, "*.csv"))):
            df = pd.read_csv(fp)
            recording_id = os.path.basename(fp)[:-4]
            try:
                feats = extract_features_from_df(df)
            except ValueError as e:
                skipped.append((fp, str(e)))
                continue
            rows.append({"Class": df["Class"].iloc[0], "Recording_ID": recording_id, **feats})
    features = pd.DataFrame(rows)
    out_dir = os.path.join(ROOT, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    features.to_csv(os.path.join(out_dir, "features_recording_level.csv"), index=False)
    model_columns = ["Class", "Recording_ID"] + [
        column for column in features.columns
        if column not in NON_MODEL_COLUMNS and column not in ("Class", "Recording_ID")
    ]
    features[model_columns].to_csv(os.path.join(out_dir, "features_clean.csv"), index=False)
    print(f"Built {len(features)} recordings x {len(model_columns) - 2} model features -> analysis/features_clean.csv")
    print(features.Class.value_counts().to_string())
    if skipped:
        print(f"\nSkipped {len(skipped)} unreadable/too-short files:")
        for fp, err in skipped[:20]:
            print(" ", os.path.basename(fp), "-", err)

if __name__ == "__main__":
    main()

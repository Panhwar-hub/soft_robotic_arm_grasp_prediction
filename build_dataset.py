"""Rebuild the feature-table CSV from all raw recordings under the three
class folders. Run this whenever new CSVs are added."""
import pandas as pd, numpy as np, glob, os
from feature_extraction import extract_features_from_df, NON_MODEL_COLUMNS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Soft_Robotic_Arm/
CLASSES = ["Ball", "Square_Box", "Triangular"]

def main():
    rows = []
    skipped = []
    for cls in CLASSES:
        for fp in sorted(glob.glob(os.path.join(ROOT, cls, "*.csv"))):
            df = pd.read_csv(fp)
            rid = os.path.basename(fp)[:-4]
            try:
                feats = extract_features_from_df(df)
            except ValueError as e:
                skipped.append((fp, str(e)))
                continue
            rows.append({"Class": df["Class"].iloc[0], "Recording_ID": rid, **feats})
    F = pd.DataFrame(rows)
    out_dir = os.path.join(ROOT, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    F.to_csv(os.path.join(out_dir, "features_recording_level.csv"), index=False)
    model_cols = ["Class", "Recording_ID"] + [c for c in F.columns if c not in NON_MODEL_COLUMNS and c not in ("Class", "Recording_ID")]
    F[model_cols].to_csv(os.path.join(out_dir, "features_clean.csv"), index=False)
    print(f"Built {len(F)} recordings x {len(model_cols)-2} model features -> analysis/features_clean.csv")
    print(F.Class.value_counts().to_string())
    if skipped:
        print(f"\nSkipped {len(skipped)} unreadable/too-short files:")
        for fp, err in skipped[:20]:
            print(" ", os.path.basename(fp), "-", err)

if __name__ == "__main__":
    main()

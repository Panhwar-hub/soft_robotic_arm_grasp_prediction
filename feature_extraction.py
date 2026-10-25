"""
Shared feature-extraction module for the Soft_Robotic_Arm capacitive-signal
classifier. Used by BOTH train_model.py and evaluate.py so training and
inference can never drift apart.

Input: one "recording" = a time series of a grasp trial with columns
  Trial_Time_s, CH0_pF, CH1_pF, CH2_pF   (same as the raw CSV files)
Output: a flat dict of features describing that recording.
"""
import numpy as np
from scipy import stats

CH = ["CH0_pF", "CH1_pF", "CH2_pF"]
BASELINE_N = 5      # samples averaged at the start of the trial to zero each channel
SPEC_GRID = 64       # points used for the time-normalised spectral analysis


def extract_features(t, ch0, ch1, ch2):
    """
    Compute the recording-level feature dict from raw arrays.

    t, ch0, ch1, ch2 : 1-D array-like, all the same length, >= ~10 samples.
    t must be increasing trial time in seconds (does not need to start at 0).

    Returns: dict[str, float]
    """
    t = np.asarray(t, dtype=float)
    raw = {"CH0_pF": np.asarray(ch0, dtype=float),
           "CH1_pF": np.asarray(ch1, dtype=float),
           "CH2_pF": np.asarray(ch2, dtype=float)}
    if len(t) < 8:
        raise ValueError(f"Recording too short ({len(t)} samples); need at least 8.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Trial_Time_s must be strictly increasing.")

    T = t[-1] - t[0]
    tn = (t - t[0]) / T  # normalised time in [0, 1]

    f = {"n_samples": len(t), "duration_s": T}

    Y = {}
    for c in CH:
        f[f"{c}_baseline_abs"] = raw[c][:BASELINE_N].mean()
        Y[c] = raw[c] - raw[c][:BASELINE_N].mean()

    for c in CH:
        y = Y[c]
        k = c[:3]
        d = np.gradient(y, t)
        i_min, i_max = y.argmin(), y.argmax()
        f.update({
            f"{k}_min": y.min(), f"{k}_max": y.max(), f"{k}_p2p": np.ptp(y),
            f"{k}_mean": y.mean(), f"{k}_std": y.std(), f"{k}_median": np.median(y),
            f"{k}_skew": stats.skew(y), f"{k}_kurt": stats.kurtosis(y),
            f"{k}_end": y[-3:].mean(),
            f"{k}_argmin_n": tn[i_min], f"{k}_argmax_n": tn[i_max],
            f"{k}_t_min_s": t[i_min] - t[0], f"{k}_t_max_s": t[i_max] - t[0],
            f"{k}_area_n": np.trapz(y, tn), f"{k}_energy": np.mean(y ** 2),
            f"{k}_dmax": d.max(), f"{k}_dmin": d.min(), f"{k}_dabs_mean": np.abs(d).mean(),
            f"{k}_zc_rate": ((y[:-1] - y.mean()) * (y[1:] - y.mean()) < 0).mean(),
            f"{k}_roughness": np.std(np.diff(y, 2)) if len(y) > 2 else 0.0,
        })
        for q, (a, b) in enumerate([(0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01)]):
            m = (tn >= a) & (tn < b)
            f[f"{k}_ph{q + 1}_mean"] = y[m].mean() if m.any() else y.mean()
        dy = np.diff(y)
        ddy = np.diff(dy)
        mob = np.sqrt(dy.var() / y.var()) if y.var() > 0 else 0.0
        f[f"{k}_hjorth_mob"] = mob
        f[f"{k}_hjorth_cmp"] = (np.sqrt(ddy.var() / dy.var()) / mob) if (dy.var() > 0 and mob > 0) else 0.0
        g = np.interp(np.linspace(0, 1, SPEC_GRID), tn, y)
        g = g - g.mean()
        P = np.abs(np.fft.rfft(g * np.hanning(SPEC_GRID))) ** 2
        fr = np.arange(len(P))
        f[f"{k}_spec_centroid"] = (fr * P).sum() / P.sum() if P.sum() > 0 else 0.0
        pn = P / P.sum() if P.sum() > 0 else P
        f[f"{k}_spec_entropy"] = -(pn[pn > 0] * np.log(pn[pn > 0])).sum()
        f[f"{k}_lowband_frac"] = P[1:4].sum() / P.sum() if P.sum() > 0 else 0.0

    f["r_01"] = np.corrcoef(Y["CH0_pF"], Y["CH1_pF"])[0, 1]
    f["r_02"] = np.corrcoef(Y["CH0_pF"], Y["CH2_pF"])[0, 1]
    f["r_12"] = np.corrcoef(Y["CH1_pF"], Y["CH2_pF"])[0, 1]
    for a, b in [("CH1", "CH0"), ("CH2", "CH0"), ("CH2", "CH1")]:
        f[f"ratio_{a}_{b}_p2p"] = f[f"{a}_p2p"] / (f[f"{b}_p2p"] + 1e-9)
        f[f"lag_{a}_{b}_argmax"] = f[f"{a}_argmax_n"] - f[f"{b}_argmin_n"]
    f["CH1_at_CH0min"] = Y["CH1_pF"][Y["CH0_pF"].argmin()]
    f["CH2_at_CH0min"] = Y["CH2_pF"][Y["CH0_pF"].argmin()]
    f["CH12_diff_mean"] = (Y["CH2_pF"] - Y["CH1_pF"]).mean()
    f["CH12_sum_mean"] = (Y["CH2_pF"] + Y["CH1_pF"]).mean()
    return f


# Different firmware/logging versions have used different column names for
# the same three physical sensor channels. List each known alias set in the
# order (channel0, channel1, channel2); the first fully-matching set wins.
CHANNEL_COLUMN_ALIASES = [
    ("CH0_pF", "CH1_pF", "CH2_pF"),
    ("Sensor1_pF", "Sensor2_pF", "Sensor3_pF"),
    ("PressureSensor1_CH0_pF", "PressureSensor2_CH1_pF", "PressureSensor3_CH2_pF"),
]


def _find_channel_columns(df):
    for cols in CHANNEL_COLUMN_ALIASES:
        if all(c in df.columns for c in cols):
            return cols
    raise ValueError(
        f"Could not find a known set of 3 channel columns in {list(df.columns)}. "
        f"Known aliases: {CHANNEL_COLUMN_ALIASES}. Add a new alias tuple to "
        f"CHANNEL_COLUMN_ALIASES in feature_extraction.py if the sensor logger "
        f"format changed again."
    )


def extract_features_from_df(df):
    """Convenience wrapper: df must have Trial_Time_s plus one of the known
    3-channel column aliases (see CHANNEL_COLUMN_ALIASES)."""
    c0, c1, c2 = _find_channel_columns(df)
    return extract_features(df["Trial_Time_s"].values, df[c0].values,
                             df[c1].values, df[c2].values)


# Columns that must NEVER be fed to the classifier: they encode absolute
# sensor offset / recording length rather than grasp shape, and were shown
# (via a same-session confound check) to leak session identity instead of
# object identity. Kept in the dict only for diagnostics.
NON_MODEL_COLUMNS = {"n_samples", "duration_s",
                      "CH0_pF_baseline_abs", "CH1_pF_baseline_abs", "CH2_pF_baseline_abs"}

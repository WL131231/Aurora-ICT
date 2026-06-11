"""FSD-style 학습 프로토타입 — setup 특징 → 승률 모델, 룰 게이트와 비교.

IN(과거 3국면) 학습 → OUT(최근 2국면) 평가 (시간 분리 = 과적합 방지).
비교: (a) 현행 룰(등급4+RR2.5+align 방향) vs (b) 모델 게이트(같은 빈도)
vs (c) 모델 최적 임계 — 총 R(승=+rr, 패=-1) 기준.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def load() -> pd.DataFrame:
    files = glob.glob("data/ml/*_*.csv")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"샘플 {len(df)} (in={len(df[df.grp == 'in'])}, out={len(df[df.grp == 'out'])})")
    return df


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame({
        "score": df["score"],
        "rr": df["rr"],
        "dir_long": df["dir_long"],
        "sl_dist_pct": df["sl_dist_pct"],
        "align_score": df["align_score"],
        "align_valid": df["align_valid"],
        "align_dir_match": np.where(
            df["dir_long"] == 1, df["align_score"], -df["align_score"],
        ),
        "hour_sin": np.sin(2 * np.pi * df["hour_utc"] / 24),
        "hour_cos": np.cos(2 * np.pi * df["hour_utc"] / 24),
        "bars_since": df["bars_since"],
    })
    for col in ("window", "source", "sym"):
        d = pd.get_dummies(df[col], prefix=col)
        x = pd.concat([x, d], axis=1)
    return x


def rule_mask(df: pd.DataFrame) -> pd.Series:
    """현행 #EDGE-V2 룰 근사 — 등급4 + RR2.5 + align 방향 일치(|점수|>=2)."""
    align_ok = (df["align_valid"] == 1) & (
        ((df["dir_long"] == 1) & (df["align_score"] >= 2))
        | ((df["dir_long"] == 0) & (df["align_score"] <= -2))
    )
    return (df["score"] >= 4) & (df["rr"] >= 2.5) & align_ok


def main() -> int:
    df = load()
    x = featurize(df)
    y = df["label_win"]
    tr = df["grp"] == "in"
    te = df["grp"] == "out"
    xtr, ytr = x[tr], y[tr]
    xte, yte = x[te], y[te]
    print(f"train 승률 base={ytr.mean():.3f} / test={yte.mean():.3f}")

    models = {
        "logreg": LogisticRegression(max_iter=2000),
        "gbdt": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0,
        ),
    }
    rule_te = rule_mask(df[te])
    r_te = df.loc[te, "r_result"].to_numpy()
    n_rule = int(rule_te.sum())
    rule_R = r_te[rule_te.to_numpy()].sum()
    print(f"\n[룰 baseline OUT] n={n_rule} 총R={rule_R:+.1f}")

    for name, m in models.items():
        m.fit(xtr.to_numpy(), ytr.to_numpy())
        p = m.predict_proba(xte.to_numpy())[:, 1]
        auc = roc_auc_score(yte, p)
        print(f"\n[{name}] test AUC={auc:.3f}")
        # (b) 룰과 같은 빈도로 자르기
        if n_rule > 0:
            thr = np.sort(p)[-n_rule]
            take = p >= thr
            print(f"  같은빈도(n={int(take.sum())}): 총R={r_te[take].sum():+.1f}"
                  f" (룰 {rule_R:+.1f})")
        # (c) 임계 sweep — 상위 빈도별 총R
        for q in (0.05, 0.10, 0.20, 0.30):
            k = max(1, int(len(p) * q))
            thr = np.sort(p)[-k]
            take = p >= thr
            print(f"  top{int(q * 100):2d}% (n={int(take.sum())}): "
                  f"총R={r_te[take].sum():+.1f} 승률={yte[take].mean():.3f}")
        if name == "gbdt":
            imp = sorted(
                zip(x.columns, m.feature_importances_, strict=False),
                key=lambda t: -t[1],
            )[:8]
            print("  중요 특징:", [(c, round(v, 3)) for c, v in imp])
    print("DONE")
    return 0


if __name__ == "__main__":
    return_code = main()
    raise SystemExit(return_code)

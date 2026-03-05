"""
Advanced Model Training & Evaluation:

  - base learners, trained on train set:
    - Gradient Boosting       
    - Random Forest            
    - Logistic Regression      
    - MLP  

  - meta-learner, trained on val set OOF predictions:
    - Logistic Regression

  - Calibration:
    - Isotonic regression on val set -> calibrated probabilities

  Alert threshold:
    - Swept on val set to maximize F1 (or F-beta)

"""

# Imports and setup
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    f1_score, precision_score, recall_score, brier_score_loss,
    confusion_matrix, classification_report,
)
from scipy.special import expit

from generate_dataset import generate_dataset
from features import build_features, temporal_train_val_test_split, METRICS


# Config
W         = 60    # look-back window
H         = 15    # prediction horizon
BETA      = 2.0   # F-beta for threshold selection (beta>1 → favor recall)
N_SAMPLES = 15000
N_INCIDENTS = 50


# Base learners
def make_base_learners():
    return {
        "GradientBoosting": Pipeline([
            ("scaler", RobustScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=300, max_depth=4,
                learning_rate=0.04, subsample=0.75,
                min_samples_leaf=10, max_features=0.7,
                random_state=123,
            )),
        ]),

        "RandomForest": Pipeline([
            ("scaler", RobustScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=10,
                class_weight="balanced",
                min_samples_leaf=5, max_features="sqrt",
                random_state=123, n_jobs=-1,
            )),
        ]),

        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.1, class_weight="balanced",
                max_iter=2000, solver="lbfgs", random_state=123,
            )),
        ]),

        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu", alpha=0.01,
                learning_rate_init=0.001,
                max_iter=300, random_state=123,
                early_stopping=True, validation_fraction=0.1,
            )),
        ]),
    }


# Stacking meta-learner
class StackedEnsemble:
    def __init__(self, base_learners: dict, beta: float = 2.0):
        self.base_learners = base_learners
        self.meta = LogisticRegression(C=1.0, max_iter=1000, random_state=123)
        self.beta  = beta
        self.threshold = 0.5
        self._names = list(base_learners.keys())

    def fit(self, X_train, y_train, X_val, y_val):
        # Train base learners on train set
        for name, pipe in self.base_learners.items():
            pipe.fit(X_train, y_train)
            print(f"{name} trained")

        # Build val meta-features
        meta_val = self._meta_features(X_val)
        self.meta.fit(meta_val, y_val)


        # Calibrate meta-learner on val set
        self.calibrator = _IsotonicCalibrator()
        raw_proba = self.meta.predict_proba(meta_val)[:, 1]
        self.calibrator.fit(raw_proba, y_val)

        # Threshold selection on val set using F-beta
        cal_proba = self.calibrator.predict(raw_proba)
        self.threshold = _select_threshold_fbeta(cal_proba, y_val, self.beta)

        print(f"threshold (F{self.beta}): {self.threshold:.3f}")

    def predict_proba(self, X):
        meta_X = self._meta_features(X)
        raw = self.meta.predict_proba(meta_X)[:, 1]

        return self.calibrator.predict(raw)

    def predict(self, X, threshold=None):
        t = threshold if threshold is not None else self.threshold
        return (self.predict_proba(X) >= t).astype(int)

    def _meta_features(self, X):
        return np.column_stack([
            pipe.predict_proba(X)[:, 1]
            for pipe in self.base_learners.values()
        ])

    def base_probas(self, X):

        return {name: pipe.predict_proba(X)[:, 1]
                for name, pipe in self.base_learners.items()}



# Simple isotonic calibration
class _IsotonicCalibrator:
    def fit(self, scores, labels):
        from sklearn.isotonic import IsotonicRegression
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(scores, labels)

    def predict(self, scores):
        return self._iso.predict(scores)



# Threshold selection
def _select_threshold_fbeta(proba, y_true, beta=2.0):
    """Select threshold maximizing f-beta on given split"""
    prec, rec, thresholds = precision_recall_curve(y_true, proba)
    fbeta = ((1 + beta**2) * prec * rec) / ((beta**2 * prec) + rec + 1e-8)
    best_idx = np.argmax(fbeta)

    if best_idx < len(thresholds):
        return thresholds[best_idx]
    
    return 0.5






#######################################################################################
####################################### Analysis, plots, etc ##########################
#######################################################################################

# to analyze different metrics across thresholds 
def threshold_sweep(proba, y_true, beta=2.0, n=60):

    thresholds = np.linspace(0.05, 0.95, n)
    rows = []

    for t in thresholds:
        pred = (proba >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f1_ = f1_score(y_true, pred, zero_division=0)
        fb = ((1 + beta**2) * p * r) / ((beta**2 * p) + r + 1e-8)

        rows.append({"threshold": t, "precision": p, "recall": r,
                     "f1": f1_, f"f{beta:.0f}": fb,
                     "n_alerts": int(pred.sum())})
        
    return pd.DataFrame(rows)


# Plotting
def plot_all(model, y_val, y_test, df, indices_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    proba_test  = model.predict_proba(X_test)
    base_probas = model.base_probas(X_test)

    # 1. Precisin-recall curves: base learners vs stacked
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ["blue", "green", "orange", "purple", "red"]

    for (name, proba), color in zip(base_probas.items(), colors):
        p, r, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        axes[0].plot(r, p, label=f"{name} (AP={ap:.3f})", lw=1.5, color=color)


    p, r, _ = precision_recall_curve(y_test, proba_test)
    ap = average_precision_score(y_test, proba_test)
    axes[0].plot(r, p, label=f"Stacked (AP={ap:.3f})", lw=2.5, color="black", ls="--")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title("PR Curves - Base vs Stacked", fontweight="bold")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[0].set_xlim([0, 1]); axes[0].set_ylim([0, 1.05])

    # 2. Threshold sweep
    sweep = threshold_sweep(proba_test, y_test, beta=BETA)
    ax2 = axes[1]
    ax2.plot(sweep["threshold"], sweep["f1"], label="F1", color="blue", lw=2)
    ax2.plot(sweep["threshold"], sweep[f"f{BETA:.0f}"], label=f"F{BETA:.0f}",
             color="red", lw=2)
    ax2.plot(sweep["threshold"], sweep["precision"], label="Precision", color="blue", lw=1.5, ls="--")
    ax2.plot(sweep["threshold"], sweep["recall"],    label="Recall",    color="green", lw=1.5, ls="--")

    
    ax2.axvline(model.threshold, color="gray", ls=":", lw=2, label=f"θ*={model.threshold:.2f}")
    ax2.set_xlabel("Alert Threshold θ"); ax2.set_ylabel("Score")
    ax2.set_title(f"Threshold Sweep (Stacked Ensemble)", fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    ax2.set_xlim([0.05, 0.95]); ax2.set_ylim([0, 1.05])

    # 3. Calibration plot
    ax3 = axes[2]
    fraction_pos, mean_pred = calibration_curve(y_test, proba_test, n_bins=15)
    ax3.plot(mean_pred, fraction_pos, "s-", color="red", label="Stacked (calibrated)", lw=2)

    # uncalibrated) meta proba
    raw_meta = model.meta.predict_proba(model._meta_features(X_test))[:, 1]
    fp_raw, mp_raw = calibration_curve(y_test, raw_meta, n_bins=15)

    ax3.plot(mp_raw, fp_raw, "^--", color="purple", label="Meta-LR (uncalibrated)", lw=1.5, alpha=0.7)
    ax3.plot([0, 1], [0, 1], "k:", lw=1.5, label="Perfect calibration")

    ax3.set_xlabel("Mean Predicted Probability")
    ax3.set_ylabel("Fraction Positives")
    ax3.set_title("Calibration Curve", fontweight="bold")
    ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/pr_threshold_calibration.png", dpi=150)
    plt.close(fig)

    # 4. Time-series view with predicted probabilities
    fig, axes2 = plt.subplots(5, 1, figsize=(20, 14), sharex=True)
    N_PLOT = min(4000, len(df))
    t  = df["timestamp"].values[:N_PLOT]
    ts = range(N_PLOT)

    metric_colors = ["blue", "green", "red", "orange"]

    for ax, metric, color in zip(axes2[:4], METRICS, metric_colors):
        ax.plot(ts, df[metric].values[:N_PLOT], lw=0.7, color=color, alpha=0.9)
        ax.set_ylabel(metric, fontsize=9)
        inc = df["incident_label"].values[:N_PLOT]
        chg = np.diff(np.concatenate([[0], inc, [0]]))
        for s, e in zip(np.where(chg == 1)[0], np.where(chg == -1)[0]):
            ax.axvspan(s, e, alpha=0.2, color="red",
                       label="Incident" if s == np.where(chg == 1)[0][0] else "")
        ax.grid(alpha=0.15)

    # Bottom: stacked ensemble probability
    ax_p = axes2[4]
    test_in_range = indices_test[indices_test < N_PLOT]
    n_plot_pts = len(test_in_range)
    if n_plot_pts > 0:
        p_plot = proba_test[:n_plot_pts]
        ax_p.plot(test_in_range, p_plot, lw=0.9, color="purple", alpha=0.85, label="P(incident)")
        ax_p.axhline(model.threshold, color="darkorange", ls="--", lw=1.5,
                     label=f"θ*={model.threshold:.2f}")
        ax_p.fill_between(test_in_range, p_plot, model.threshold,
                          where=(p_plot >= model.threshold),
                          alpha=0.35, color="darkorange", label="Alert fired")
    ax_p.set_ylabel("Pred. Prob.", fontsize=9); ax_p.set_ylim([0, 1])
    ax_p.legend(fontsize=9, loc="upper right"); ax_p.grid(alpha=0.15)
    ax_p.set_xlabel("Time step (minutes)", fontsize=10)

    axes2[0].set_title("Multivariate Metrics with Incident Labels & Stacked Ensemble Predictions",
                        fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/timeseries_stacked.png", dpi=150)
    plt.close(fig)

    # 5. Feature importances (GBM)
    gbm = model.base_learners["GradientBoosting"].named_steps["clf"]
    imp = gbm.feature_importances_
    top_idx = np.argsort(imp)[::-1][:25]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(25), imp[top_idx[::-1]], color="#5C6BC0", alpha=0.85)
    ax.set_yticks(range(25))
    ax.set_yticklabels([col_names[i] for i in top_idx[::-1]], fontsize=8)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title("Top-25 Feature Importances (GradientBoosting base learner)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/feature_importances.png", dpi=150)
    plt.close(fig)


    return sweep

# Main

if __name__ == "__main__":
    OUTPUT_DIR = "."

    # 1. Generate dataset
    print("\nGenerating dataset…")
    df = generate_dataset(n_samples=N_SAMPLES, n_incidents=N_INCIDENTS, seed=123)
    print(f" dataset generated {df.shape}")

    # 2. Build features
    print(f"\n Building features…")
    X, y, indices, col_names = build_features(df, METRICS, W=W, H=H, step=1)
    print(f"    Feature matrix: {X.shape}")

    # Check for NaN / Inf
    bad = ~np.isfinite(X)
    if bad.any():
        print(f"{bad.sum()} non-finite values - replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Split
    print("\nData splitting...")
    (X_train, y_train, idx_train,
     X_val,   y_val,   idx_val,
     X_test,  y_test,  idx_test) = temporal_train_val_test_split(
        X, y, indices, val_ratio=0.10, test_ratio=0.20
    )
    print(f"Train: {len(X_train)}")
    print(f"Val:   {len(X_val)}")
    print(f"Test:  {len(X_test)}")

    # 4. Train stacked ensemble
    print("\nTraining stacked ensemble…")
    model = StackedEnsemble(make_base_learners(), beta=BETA)
    model.fit(X_train, y_train, X_val, y_val)
    print("Trained")

    # 5. Evaluation
    print("\nEvaluation...")
    proba_test = model.predict_proba(X_test)
    pred_test  = model.predict(X_test)   # we use calibrated threshold

    pr_auc  = average_precision_score(y_test, proba_test)
    roc_auc = roc_auc_score(y_test, proba_test)
    brier   = brier_score_loss(y_test, proba_test)
    f1      = f1_score(y_test, pred_test, zero_division=0)
    prec    = precision_score(y_test, pred_test, zero_division=0)
    rec     = recall_score(y_test, pred_test, zero_division=0)

    print(f"\nPR-AUC: {pr_auc:.4f}  (primary)")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Brier: {brier:.4f}  (calibration quality, lower=better)")
    print(f"F1: {f1:.4f}  (θ*={model.threshold:.3f})")
    print(f"F{BETA:.0f}      : {((1+BETA**2)*prec*rec)/((BETA**2*prec)+rec+1e-8):.4f}")
    print(f"Prec: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print()
    print(classification_report(y_test, pred_test, target_names=["Normal", "Incident"]))

    # Base learner comparison
    print("\n  Base learner comparison (test set):")
    print(f"  {'Model':<22} {'PR-AUC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    print("  " + "-"*55)
    for name, proba_b in model.base_probas(X_test).items():
        pred_b = (proba_b >= model.threshold).astype(int)
        print(f"  {name:<22} "
              f"{average_precision_score(y_test, proba_b):>8.4f} "
              f"{f1_score(y_test, pred_b, zero_division=0):>8.4f} "
              f"{precision_score(y_test, pred_b, zero_division=0):>8.4f} "
              f"{recall_score(y_test, pred_b, zero_division=0):>8.4f}")
    print(f"  {'Stacked (calibrated)':<22} "
          f"{pr_auc:>8.4f} {f1:>8.4f} {prec:>8.4f} {rec:>8.4f}")

    # 6. Plots
    print("\nGenerating plots and saving threshold sweep…")
    sweep = plot_all(model,y_val, y_test, df, idx_test, OUTPUT_DIR)
    sweep.to_csv(f"{OUTPUT_DIR}/threshold_sweep.csv", index=False)
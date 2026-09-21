"""
evaluate_hybrid_vs_classical.py
================================
Fair, apples-to-apples comparison of FOUR modeling arms on the exact same
repeated-stratified-CV folds:

  1. classical_only  - a classical model (default: RandomForest) on your
                       15 LASSO-selected features. SWAP IN YOUR OWN TUNED
                       MODEL/PIPELINE in build_classical_model() below if
                       you want this arm to reproduce your real 88% result
                       - otherwise this is just a reasonable default, not
                       guaranteed to match it.
  2. quantum_only    - QSVM (quantum-kernel SVM), same as train_quantum_only.py:
                       15 LASSO features -> PCA to N_QUBITS_QUANTUM -> quantum
                       kernel -> classical SVM.
  3. hybrid          - classical features CONCATENATED with quantum-kernel-
                       derived features (via Quantum Kernel PCA, now
                       N_HYBRID_QFEATURES=8 components), re-pruned by LASSO
                       down to N_FEATURES_FINAL_HYBRID=20 (widened from 15,
                       so quantum features aren't forced into a 1-for-1
                       fight against classical ones to survive), fed to the
                       SAME classical model as arm 1.
  4. ensemble_avg    - a DIFFERENT fusion strategy: instead of merging
                       features before training, fit classical_only and
                       quantum_only independently (already done for arms 1
                       and 2) and average their predicted probabilities.
                       This only helps if the two models make different
                       KINDS of errors - if their mistakes overlap, this
                       won't beat the stronger model alone, and that's a
                       legitimate, reportable finding too.

Because every arm sees the identical train/test split on every fold, you can
run a paired test (paired t-test + Wilcoxon signed-rank) across folds to see
whether any arm beating (or losing to) another is a real effect or just
fold-to-fold noise - this matters a lot with only ~15-25 folds.

Outputs, all under <csv_dir>/model_outputs_hybrid_eval/:
  - fold_level_results.csv   one row per (dataset, strategy, fold, arm)
  - summary_results.csv      mean +/- std per (dataset, strategy, arm)
  - paired_tests.csv         hybrid vs classical, hybrid vs quantum:
                              paired t-test AND Wilcoxon signed-rank,
                              on both accuracy and F1

HONESTY NOTE: this script does not know what "should" happen. If hybrid
comes out worse than classical-only or quantum-only on your data, that is a
real, reportable result - do not discard it. A rigorous "we tried X, it did
not help, here is why" is stronger evidence for a review/judging panel than
a suspiciously clean win.

Run:
    python evaluate_hybrid_vs_classical.py
(edit CSV_PATH / CLINICAL_XLSX_PATH below first)
"""

import os
import sys
import time
import traceback
import warnings
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    try:
        from qiskit_aer.primitives import SamplerV2 as AerSampler
        HAS_AER = True
    except ImportError:
        HAS_AER = False
    try:
        from qiskit_machine_learning.state_fidelities import ComputeUncompute
    except ImportError:
        from qiskit_algorithms.state_fidelities import ComputeUncompute
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    HAS_AER = False

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG - edit these
# =============================================================================
CSV_PATH = r"D:\ABDM\Compressed\als_voice_features_with_literature_feats.csv"
CLINICAL_XLSX_PATH = r"D:\ABDM\Compressed\clinical_data.xlsx"

N_SPLITS = 5
N_REPEATS = 3          # 15 folds total; raise once you've confirmed it runs
RANDOM_STATE = 42
N_JOBS = max(1, (os.cpu_count() or 4) - 1)

N_FEATURES_FINAL = 15          # classical-arm LASSO feature count
N_QUBITS_QUANTUM = 4           # PCA components fed to the quantum circuit
QUANTUM_REPS = 2
N_HYBRID_QFEATURES = 8         # quantum-kernel-PCA components added to the classical set
                                # (raised from 4: more room for the kernel to express structure)
N_FEATURES_FINAL_HYBRID = 20   # re-pruned via LASSO from (15 classical + 8 quantum = 23) candidates
                                # (raised from 15: quantum features no longer have to displace a
                                # classical one 1-for-1 to survive)

DYSARTHRIC_CUTOFF = 2
ID_COLS = ["filename", "filepath", "label", "subject_id", "task"]

ESSENTIAL_RAW_FEATURES = [
    "PPE", "F0_std", "F0_min", "F0_CV",
    "HNR_mean", "HNR_std", "NHR", "GNE", "CPP",
    "Shimmer_APQ11", "Shimmer_DDA", "Shimmer_local_percent",
    "DFA", "RPDE", "Spread1", "Spread2",
    "PVI",
    "RelH2", "RelH7", "RelH8", "H2_std", "H7_std", "H8_mean",
    "MFCC_1_mean", "MFCC_8_mean",
]


def build_classical_model():
    """*** EDIT THIS to plug in the real model/pipeline that hit 88% for you ***
    Whatever you put here is used identically for arm 1 (classical-only) and
    arm 3 (hybrid), so the comparison stays apples-to-apples."""
    return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=RANDOM_STATE)


# =============================================================================
# Data loading (same logic as train_quantum_only.py)
# =============================================================================
def load_clinical_data(xlsx_path: Path):
    sheets = pd.ExcelFile(xlsx_path).sheet_names
    sheet = "VOC-ALS_Data" if "VOC-ALS_Data" in sheets else 0
    probe = pd.read_excel(xlsx_path, sheet_name=sheet, header=None, nrows=2)
    row0_is_placeholder = probe.iloc[0].astype(str).str.match(r"^Column-\d+$").mean() > 0.8
    header_row = 1 if row0_is_placeholder else 0
    clin = pd.read_excel(xlsx_path, sheet_name=sheet, header=header_row)
    clin.columns = [str(c).strip() for c in clin.columns]
    id_col = next((c for c in clin.columns if c.strip().lower() in ("id", "subject_id", "subjectid")), None)
    speech_col = next((c for c in clin.columns if "speech" in c.lower() and "subscore" in c.lower()), None)
    if id_col is None:
        raise ValueError(f"Couldn't find an ID column. Columns found: {list(clin.columns)}")
    clin = clin.rename(columns={id_col: "subject_id"})
    if speech_col:
        clin = clin.rename(columns={speech_col: "speech_subscore"})
        clin["speech_subscore"] = pd.to_numeric(clin["speech_subscore"], errors="coerce")
    return clin, speech_col is not None


def aggregate_to_subject_level(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in ID_COLS]
    agg = df.groupby("subject_id")[feature_cols].agg(["mean", "std"])
    agg.columns = [f"{feat}_{stat}" for feat, stat in agg.columns]
    agg = agg.reset_index()
    labels = df.groupby("subject_id")["label"].first().reset_index()
    subj_df = agg.merge(labels, on="subject_id")
    std_cols = [c for c in subj_df.columns if c.endswith("_std")]
    subj_df[std_cols] = subj_df[std_cols].fillna(0)
    feature_cols_agg = [c for c in subj_df.columns if c not in ("subject_id", "label")]
    return subj_df, feature_cols_agg


def curated_pool_columns(all_feature_cols):
    pool = []
    for raw in ESSENTIAL_RAW_FEATURES:
        for suffix in ("_mean", "_std"):
            col = f"{raw}{suffix}"
            if col in all_feature_cols:
                pool.append(col)
    return pool


def select_features_lasso(X_train, y_train, feature_names, n_final):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    lasso = LassoCV(cv=3, random_state=RANDOM_STATE, max_iter=10000)
    lasso.fit(X_scaled, y_train)
    importance = np.abs(lasso.coef_)
    order = np.argsort(importance)[::-1]
    return [feature_names[i] for i in order[:n_final]]


def _flatten(circuit):
    prev_ops = None
    for _ in range(6):
        ops = {instr.operation.name for instr in circuit.data}
        if ops == prev_ops:
            break
        prev_ops = ops
        circuit = circuit.decompose()
    return circuit


# =============================================================================
# Quantum kernel helpers (shared by arm 2 and arm 3)
# =============================================================================
def build_quantum_kernel(n_qubits):
    feature_map = _flatten(ZZFeatureMap(feature_dimension=n_qubits, reps=QUANTUM_REPS,
                                         entanglement="linear"))
    if HAS_AER:
        fidelity = ComputeUncompute(sampler=AerSampler())
        return FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    return FidelityQuantumKernel(feature_map=feature_map)


def quantum_reduce(X, pca, qscaler):
    return qscaler.transform(pca.transform(X))


def fit_qsvm(X_train_q, y_train):
    """Arm 2: pure QSVM on the quantum-reduced features."""
    kernel = build_quantum_kernel(X_train_q.shape[1])
    model = QSVC(quantum_kernel=kernel, probability=False)
    model.fit(X_train_q, y_train)
    return model


def fit_quantum_kernel_pca(X_train_q, n_components):
    """Fits a Quantum-Kernel-PCA: computes the train-train quantum kernel
    (fidelity) matrix, then runs classical KernelPCA(kernel='precomputed')
    on top of it to get a small set of quantum-derived features."""
    kernel = build_quantum_kernel(X_train_q.shape[1])
    K_train = kernel.evaluate(X_train_q)
    kpca = KernelPCA(n_components=n_components, kernel="precomputed", random_state=RANDOM_STATE)
    Z_train = kpca.fit_transform(K_train)
    return kernel, kpca, Z_train


def transform_quantum_kernel_pca(X_new_q, X_train_q, kernel, kpca):
    K_new = kernel.evaluate(X_new_q, X_train_q)
    return kpca.transform(K_new)


# =============================================================================
# Per-fold evaluation: all three arms on the identical split
# =============================================================================
def _proba_like(model, X):
    """Returns a probability-like score in [0,1] for the positive class,
    whatever the model supports. QSVC is fit with probability=False (see
    fit_qsvm - probability=True triggers a ~5x-slower internal calibration),
    so it has no predict_proba; decision_function + a sigmoid gives a usable,
    monotonic substitute for averaging/roc_auc purposes."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        from scipy.special import expit
        return expit(model.decision_function(X))
    return None


def _metrics(y_test, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": (roc_auc_score(y_test, y_proba)
                    if y_proba is not None and len(np.unique(y_test)) > 1 else np.nan),
    }


def _run_one_fold(fold_num, total_folds, train_idx, test_idx, X_pool, y, pool_names,
                   dataset_name, strategy_name):
    fold_start = time.time()
    X_train, X_test = X_pool[train_idx], X_pool[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if min(np.bincount(y_train)) < 2:
        print(f"[progress] fold {fold_num}/{total_folds} ({dataset_name}/{strategy_name}) - "
              f"skipped (minority class too small)", flush=True)
        return []

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    rows = []

    # ---- Arm 1: classical-only ----------------------------------------
    t0 = time.time()
    classical_feats = select_features_lasso(X_train_imp, y_train, pool_names, N_FEATURES_FINAL)
    idx_c = [pool_names.index(f) for f in classical_feats]
    scaler_c = StandardScaler()
    Xc_train = scaler_c.fit_transform(X_train_imp[:, idx_c])
    Xc_test = scaler_c.transform(X_test_imp[:, idx_c])

    clf = clone(build_classical_model())
    clf.fit(Xc_train, y_train)
    y_pred = clf.predict(Xc_test)
    y_proba_classical = _proba_like(clf, Xc_test)
    rows.append({"fold": fold_num, "arm": "classical_only", **_metrics(y_test, y_pred, y_proba_classical)})
    print(f"[progress] fold {fold_num}/{total_folds} classical_only: done in {time.time()-t0:.1f}s "
          f"(acc={rows[-1]['accuracy']:.3f})", flush=True)

    # ---- Prep quantum-reduced features (shared by arm 2 & 3) -----------
    t0 = time.time()
    n_comp = min(N_QUBITS_QUANTUM, X_train_imp.shape[1], X_train_imp.shape[0])
    pca_q = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    Xq_train_reduced = pca_q.fit_transform(X_train_imp)
    Xq_test_reduced = pca_q.transform(X_test_imp)
    qscaler = MinMaxScaler(feature_range=(0, np.pi))
    Xq_train = qscaler.fit_transform(Xq_train_reduced)
    Xq_test = qscaler.transform(Xq_test_reduced)

    # ---- Arm 2: quantum-only (QSVM) ------------------------------------
    try:
        qsvm = fit_qsvm(Xq_train, y_train)
        y_pred = np.asarray(qsvm.predict(Xq_test))
        y_proba_quantum = _proba_like(qsvm, Xq_test)
        rows.append({"fold": fold_num, "arm": "quantum_only", **_metrics(y_test, y_pred, y_proba_quantum)})
        print(f"[progress] fold {fold_num}/{total_folds} quantum_only: done in {time.time()-t0:.1f}s "
              f"(acc={rows[-1]['accuracy']:.3f})", flush=True)

        # ---- Arm 4: probability-averaging ensemble ---------------------
        # A different fusion strategy than concatenating features before
        # training: fit classical and quantum models independently (already
        # done above), then average their predicted probabilities. This only
        # helps if the two models make DIFFERENT kinds of errors - if their
        # mistakes overlap, averaging won't beat the stronger model alone.
        if y_proba_quantum is not None and y_proba_classical is not None:
            y_proba_ens = 0.5 * y_proba_classical + 0.5 * y_proba_quantum
            y_pred_ens = (y_proba_ens >= 0.5).astype(int)
            rows.append({"fold": fold_num, "arm": "ensemble_avg",
                         **_metrics(y_test, y_pred_ens, y_proba_ens)})
            print(f"[progress] fold {fold_num}/{total_folds} ensemble_avg: "
                  f"(acc={rows[-1]['accuracy']:.3f})", flush=True)
    except Exception as e:
        print(f"[progress] fold {fold_num}/{total_folds} quantum_only: FAILED "
              f"({type(e).__name__}: {e})", flush=True)

    # ---- Arm 3: hybrid (classical features + quantum-kernel-PCA feats) -
    t0 = time.time()
    try:
        kernel, kpca, Zq_train = fit_quantum_kernel_pca(Xq_train, N_HYBRID_QFEATURES)
        Zq_test = transform_quantum_kernel_pca(Xq_test, Xq_train, kernel, kpca)

        qfeat_names = [f"qkpca_{i}" for i in range(Zq_train.shape[1])]
        hybrid_pool_names = classical_feats + qfeat_names
        Xh_train_full = np.hstack([X_train_imp[:, idx_c], Zq_train])
        Xh_test_full = np.hstack([X_test_imp[:, idx_c], Zq_test])

        # Re-run LASSO over (classical + quantum) candidates: the quantum
        # features only survive into the final model if they earn their place.
        hybrid_selected = select_features_lasso(Xh_train_full, y_train, hybrid_pool_names,
                                                 min(N_FEATURES_FINAL_HYBRID, len(hybrid_pool_names)))
        idx_h = [hybrid_pool_names.index(f) for f in hybrid_selected]
        n_q_kept = sum(1 for f in hybrid_selected if f.startswith("qkpca_"))

        scaler_h = StandardScaler()
        Xh_train = scaler_h.fit_transform(Xh_train_full[:, idx_h])
        Xh_test = scaler_h.transform(Xh_test_full[:, idx_h])

        clf_h = clone(build_classical_model())
        clf_h.fit(Xh_train, y_train)
        y_pred = clf_h.predict(Xh_test)
        y_proba = clf_h.predict_proba(Xh_test)[:, 1] if hasattr(clf_h, "predict_proba") else None
        rows.append({"fold": fold_num, "arm": "hybrid", "n_quantum_features_kept": n_q_kept,
                     **_metrics(y_test, y_pred, y_proba)})
        print(f"[progress] fold {fold_num}/{total_folds} hybrid: done in {time.time()-t0:.1f}s "
              f"(acc={rows[-1]['accuracy']:.3f}, kept {n_q_kept}/{len(qfeat_names)} quantum feats)",
              flush=True)
    except Exception as e:
        print(f"[progress] fold {fold_num}/{total_folds} hybrid: FAILED "
              f"({type(e).__name__}: {e})", flush=True)
        traceback.print_exc()

    print(f"[progress] fold {fold_num}/{total_folds} complete in {time.time()-fold_start:.1f}s", flush=True)
    for r in rows:
        r["dataset"] = dataset_name
        r["feature_strategy"] = strategy_name
    return rows


def run_repeated_cv(X, y, feature_names, feature_pool, strategy_name, dataset_name):
    pool_idx = [feature_names.index(f) for f in feature_pool]
    X_pool = X[:, pool_idx]
    pool_names = feature_pool

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    splits = list(rskf.split(X_pool, y))
    total_folds = len(splits)

    print(f"[info] Running {total_folds} folds across up to {N_JOBS} parallel worker(s)...", flush=True)
    run_start = time.time()
    outcomes = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_run_one_fold)(i + 1, total_folds, train_idx, test_idx, X_pool, y,
                                pool_names, dataset_name, strategy_name)
        for i, (train_idx, test_idx) in enumerate(splits)
    )
    fold_rows = [r for fold_rows in outcomes for r in fold_rows]
    elapsed = time.time() - run_start
    print(f"[info] {dataset_name}/{strategy_name}: {total_folds} folds finished in "
          f"{elapsed/60:.1f} min total", flush=True)
    return pd.DataFrame(fold_rows)


# =============================================================================
# Paired statistical tests: is the hybrid difference real, or noise?
# =============================================================================
def paired_tests(fold_df, dataset_name, strategy_name):
    out = []
    piv_acc = fold_df.pivot_table(index="fold", columns="arm", values="accuracy")
    piv_f1 = fold_df.pivot_table(index="fold", columns="arm", values="f1")
    for metric_name, piv in [("accuracy", piv_acc), ("f1", piv_f1)]:
        for a, b in [("hybrid", "classical_only"), ("hybrid", "quantum_only"),
                     ("ensemble_avg", "classical_only"), ("ensemble_avg", "quantum_only"),
                     ("ensemble_avg", "hybrid")]:
            if a not in piv.columns or b not in piv.columns:
                continue
            paired = piv[[a, b]].dropna()
            if len(paired) < 3:
                continue
            t_stat, t_p = ttest_rel(paired[a], paired[b])
            try:
                w_stat, w_p = wilcoxon(paired[a], paired[b])
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            out.append({
                "dataset": dataset_name, "feature_strategy": strategy_name, "metric": metric_name,
                "arm_a": a, "arm_b": b, "n_folds_paired": len(paired),
                "mean_a": paired[a].mean(), "mean_b": paired[b].mean(),
                "mean_diff_a_minus_b": (paired[a] - paired[b]).mean(),
                "ttest_p": t_p, "wilcoxon_p": w_p,
            })
    return out


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not HAS_QISKIT:
        print("[error] qiskit / qiskit-machine-learning not found. Install with:")
        print("    pip install qiskit qiskit-machine-learning qiskit-algorithms qiskit-aer")
        return
    print(f"[info] qiskit-aer available: {HAS_AER}")

    csv_path = Path(CSV_PATH if len(sys.argv) < 2 else sys.argv[1])
    clinical_path = Path(CLINICAL_XLSX_PATH if len(sys.argv) < 3 else sys.argv[2])
    if not csv_path.exists() or not clinical_path.exists():
        print(f"[error] Check paths: {csv_path} / {clinical_path}")
        return

    out_dir = csv_path.parent / "model_outputs_hybrid_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    feature_cols_raw = [c for c in df.columns if c not in ID_COLS]
    df = df.dropna(subset=feature_cols_raw).reset_index(drop=True)

    clin, has_severity = load_clinical_data(clinical_path)
    subj_df, feature_cols = aggregate_to_subject_level(df)
    curated_pool = curated_pool_columns(feature_cols)

    if has_severity:
        subj_df = subj_df.merge(clin[["subject_id", "speech_subscore"]], on="subject_id", how="left")

    le = LabelEncoder()
    subj_df["label_enc"] = le.fit_transform(subj_df["label"].values)
    datasets = {"A_Full_ALSvsHC": subj_df}
    if has_severity:
        is_dysarthric = (subj_df["label"] == "ALS") & (subj_df["speech_subscore"] <= DYSARTHRIC_CUTOFF)
        is_hc = subj_df["label"] == "HC"
        datasets["B_StrictDysarthricALSvsHC"] = subj_df[is_dysarthric | is_hc].reset_index(drop=True)

    all_fold_rows = []
    all_tests = []
    for dataset_name, d in datasets.items():
        y = d["label_enc"].values
        X = d[feature_cols].values
        if min(np.bincount(y)) < N_SPLITS:
            print(f"[warning] {dataset_name}: minority class too small for {N_SPLITS}-fold CV, skipping.")
            continue
        print(f"\n[info] === {dataset_name}: {len(d)} subjects ===")
        for strategy_name, pool in [("Full_214_LASSO", feature_cols),
                                     ("Curated_Pool_LASSO", curated_pool)]:
            print(f"[info] Running: {strategy_name} ({len(pool)} candidates)...", flush=True)
            try:
                fold_df = run_repeated_cv(X, y, feature_cols, pool, strategy_name, dataset_name)
            except Exception as e:
                print(f"[error] {dataset_name}/{strategy_name} crashed: {type(e).__name__}: {e}")
                traceback.print_exc()
                continue
            if fold_df.empty:
                continue
            all_fold_rows.append(fold_df)
            fold_df.to_csv(out_dir / f"checkpoint_{dataset_name}_{strategy_name}.csv", index=False)
            all_tests.extend(paired_tests(fold_df, dataset_name, strategy_name))
            print(f"[checkpoint] saved checkpoint_{dataset_name}_{strategy_name}.csv", flush=True)

    if not all_fold_rows:
        print("[error] No results produced.")
        return

    fold_level = pd.concat(all_fold_rows, ignore_index=True)
    fold_level.to_csv(out_dir / "fold_level_results.csv", index=False)

    summary = (fold_level.groupby(["dataset", "feature_strategy", "arm"])
               .agg(n_folds=("accuracy", "count"),
                    accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                    f1_mean=("f1", "mean"), f1_std=("f1", "std"),
                    precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
                    roc_auc_mean=("roc_auc", "mean"))
               .reset_index()
               .sort_values(["dataset", "feature_strategy", "f1_mean"], ascending=[True, True, False]))
    summary.to_csv(out_dir / "summary_results.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}" if isinstance(v, float) else str(v))
    print("\n" + "=" * 140)
    print("SUMMARY: classical_only vs quantum_only vs hybrid (same CV folds)")
    print("=" * 140)
    print(summary.to_string(index=False))

    if all_tests:
        tests_df = pd.DataFrame(all_tests)
        tests_df.to_csv(out_dir / "paired_tests.csv", index=False)
        print("\n" + "=" * 140)
        print("PAIRED SIGNIFICANCE TESTS (is hybrid's difference real, or fold-to-fold noise?)")
        print("p < 0.05 in EITHER column suggests a real difference on this data; higher p means")
        print("you cannot distinguish the arms with this many folds - report that honestly.")
        print("=" * 140)
        print(tests_df.to_string(index=False))

    print(f"\n[done] All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()

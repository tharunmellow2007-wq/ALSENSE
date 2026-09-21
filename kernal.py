"""
evaluate_kernel_alignment.py
=============================
Tests whether a TRAINED (data-aligned) quantum kernel closes the gap to
classical performance, versus the fixed generic ZZFeatureMap kernel used
so far.

Arms compared, on the SAME CV folds:
  1. classical_only    - your classical baseline (EDIT build_classical_model()
                          below to match your real 88%+ model)
  2. quantum_fixed      - the same fixed ZZFeatureMap QSVM used in earlier
                          scripts (kept here as a reference point)
  3. quantum_aligned    - QSVM using a TRAINED feature map: per-feature
                          rotation-scaling parameters optimized to maximize
                          kernel-target alignment (KTA / SVCLoss) BEFORE
                          fitting QSVM
  4. hybrid_aligned     - classical features concatenated with
                          Quantum-Kernel-PCA features computed from the
                          ALIGNED kernel (not the fixed one), re-pruned by
                          LASSO, fed to the classical model

WHY A SUBSAMPLE FOR TRAINING THE KERNEL:
Optimizing the kernel's parameters requires recomputing a kernel matrix on
every optimizer iteration. Doing that on your full ~55-122 sample training
fold, for ~20 iterations, would take hours per fold. Instead, the kernel is
TRAINED on a small stratified subsample (ALIGNMENT_SUBSAMPLE_SIZE, default
20), then the learned parameters are used to build the kernel on the FULL
training fold for the actual QSVM fit - same cost as your original QSVM run,
plus a comparatively cheap alignment step. This is a standard practical
compromise, not a shortcut that invalidates the result, but it does mean the
alignment signal comes from a smaller sample than the final model uses.

HONESTY NOTE: this may not help. If quantum_aligned performs about the same
as quantum_fixed, that itself is a real finding (the fixed kernel wasn't the
bottleneck - the information bottleneck of 4 qubits / small sample size is).
Report whatever actually happens.

Outputs, under <csv_dir>/model_outputs_kernel_alignment/:
  - fold_level_results.csv
  - summary_results.csv
  - paired_tests.csv   (includes quantum_aligned vs quantum_fixed - the key
                        check for whether alignment did anything at all -
                        and hybrid_aligned vs classical_only)

Run:
    python evaluate_kernel_alignment.py
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
from scipy.special import expit

from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

HAS_QISKIT = True
HAS_ALIGNMENT = True
IMPORT_ERROR_MSG = ""
try:
    from qiskit.circuit import QuantumCircuit, ParameterVector
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

    # --- kernel-alignment-specific imports: paths vary across versions ---
    try:
        from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel
    except ImportError:
        from qiskit_machine_learning.kernels.trainable_kernel import TrainableFidelityQuantumKernel

    try:
        from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
    except ImportError:
        from qiskit_machine_learning.algorithms import QuantumKernelTrainer

    try:
        from qiskit_machine_learning.utils.loss_functions import SVCLoss
    except ImportError:
        from qiskit_machine_learning.kernels.algorithms import SVCLoss

    try:
        from qiskit_algorithms.optimizers import COBYLA
    except ImportError:
        from qiskit.algorithms.optimizers import COBYLA

except ImportError as e:
    HAS_QISKIT = False
    HAS_ALIGNMENT = False
    HAS_AER = False
    IMPORT_ERROR_MSG = str(e)

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
CSV_PATH = r"D:\ABDM\Compressed\als_voice_features_with_literature_feats.csv"
CLINICAL_XLSX_PATH = r"D:\ABDM\Compressed\clinical_data.xlsx"

N_SPLITS = 5
N_REPEATS = 3
RANDOM_STATE = 42
N_JOBS = max(1, (os.cpu_count() or 4) - 1)

N_FEATURES_FINAL = 15
N_QUBITS_QUANTUM = 4
QUANTUM_REPS = 2
N_HYBRID_QFEATURES = 8
N_FEATURES_FINAL_HYBRID = 20

ALIGNMENT_SUBSAMPLE_SIZE = 20      # subsample used ONLY to train kernel parameters
KERNEL_ALIGNMENT_MAXITER = 20      # optimizer iterations for kernel training

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
    """*** EDIT THIS to plug in the real model/pipeline that hit 88% ***"""
    return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=RANDOM_STATE)


# =============================================================================
# Data loading (identical to evaluate_hybrid_vs_classical.py)
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
# Fixed (untrained) quantum kernel - reference point, same as earlier scripts
# =============================================================================
def build_fixed_kernel(n_qubits):
    feature_map = _flatten(ZZFeatureMap(feature_dimension=n_qubits, reps=QUANTUM_REPS,
                                         entanglement="linear"))
    if HAS_AER:
        return FidelityQuantumKernel(feature_map=feature_map, fidelity=ComputeUncompute(sampler=AerSampler()))
    return FidelityQuantumKernel(feature_map=feature_map)


# =============================================================================
# Trainable (alignable) quantum kernel
# =============================================================================
def build_trainable_feature_map(n_qubits, reps):
    """Per-feature rotation-scaling parameters (theta), shared across reps,
    optimized to maximize kernel-target alignment. Kept to n_qubits trainable
    parameters (not n_qubits*reps) to keep the optimization problem small and
    fast to fit on a subsample."""
    x = ParameterVector("x", n_qubits)
    theta = ParameterVector("theta", n_qubits)
    qc = QuantumCircuit(n_qubits)
    for _ in range(reps):
        for i in range(n_qubits):
            qc.ry(theta[i] * x[i], i)
        for i in range(n_qubits - 1):
            qc.cz(i, i + 1)
    return qc, list(theta)


def train_aligned_kernel(X_train_q, y_train, n_qubits):
    """Trains kernel parameters on a small stratified subsample (cheap),
    returns a TrainableFidelityQuantumKernel with the learned parameters
    already assigned, ready to .evaluate() on the full training set."""
    n_sub = min(ALIGNMENT_SUBSAMPLE_SIZE, len(X_train_q))
    if n_sub < len(X_train_q) and min(np.bincount(y_train)) >= 2:
        X_sub, _, y_sub, _ = train_test_split(
            X_train_q, y_train, train_size=n_sub, stratify=y_train, random_state=RANDOM_STATE)
    else:
        X_sub, y_sub = X_train_q, y_train

    fmap, theta_params = build_trainable_feature_map(n_qubits, QUANTUM_REPS)
    fidelity = ComputeUncompute(sampler=AerSampler()) if HAS_AER else None
    kwargs = dict(feature_map=fmap, training_parameters=theta_params)
    if fidelity is not None:
        kwargs["fidelity"] = fidelity
    kernel = TrainableFidelityQuantumKernel(**kwargs)

    loss = SVCLoss(C=1.0)
    optimizer = COBYLA(maxiter=KERNEL_ALIGNMENT_MAXITER)
    initial_point = np.ones(len(theta_params))  # start at the "fixed kernel" equivalent (scale=1)

    trainer = QuantumKernelTrainer(quantum_kernel=kernel, loss=loss, optimizer=optimizer,
                                    initial_point=initial_point)
    result = trainer.fit(X_sub, y_sub)
    return result.quantum_kernel


# =============================================================================
# Per-fold evaluation
# =============================================================================
def _proba_like(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
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
        print(f"[progress] fold {fold_num}/{total_folds} - skipped (minority class too small)", flush=True)
        return []

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    rows = []

    # ---- Arm 1: classical_only ----------------------------------------
    t0 = time.time()
    classical_feats = select_features_lasso(X_train_imp, y_train, pool_names, N_FEATURES_FINAL)
    idx_c = [pool_names.index(f) for f in classical_feats]
    scaler_c = StandardScaler()
    Xc_train = scaler_c.fit_transform(X_train_imp[:, idx_c])
    Xc_test = scaler_c.transform(X_test_imp[:, idx_c])
    clf = clone(build_classical_model())
    clf.fit(Xc_train, y_train)
    rows.append({"fold": fold_num, "arm": "classical_only",
                 **_metrics(y_test, clf.predict(Xc_test), _proba_like(clf, Xc_test))})
    print(f"[progress] fold {fold_num}/{total_folds} classical_only: {time.time()-t0:.1f}s "
          f"(acc={rows[-1]['accuracy']:.3f})", flush=True)

    # ---- Shared quantum-reduced features --------------------------------
    n_comp = min(N_QUBITS_QUANTUM, X_train_imp.shape[1], X_train_imp.shape[0])
    pca_q = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    Xq_train_reduced = pca_q.fit_transform(X_train_imp)
    Xq_test_reduced = pca_q.transform(X_test_imp)
    qscaler = MinMaxScaler(feature_range=(0, np.pi))
    Xq_train = qscaler.fit_transform(Xq_train_reduced)
    Xq_test = qscaler.transform(Xq_test_reduced)

    # ---- Arm 2: quantum_fixed (reference, same as earlier scripts) -----
    try:
        t0 = time.time()
        fixed_kernel = build_fixed_kernel(Xq_train.shape[1])
        qsvm_fixed = QSVC(quantum_kernel=fixed_kernel, probability=False)
        qsvm_fixed.fit(Xq_train, y_train)
        y_pred = np.asarray(qsvm_fixed.predict(Xq_test))
        rows.append({"fold": fold_num, "arm": "quantum_fixed",
                     **_metrics(y_test, y_pred, _proba_like(qsvm_fixed, Xq_test))})
        print(f"[progress] fold {fold_num}/{total_folds} quantum_fixed: {time.time()-t0:.1f}s "
              f"(acc={rows[-1]['accuracy']:.3f})", flush=True)
    except Exception as e:
        print(f"[progress] fold {fold_num}/{total_folds} quantum_fixed: FAILED ({type(e).__name__}: {e})",
              flush=True)

    # ---- Arm 3: quantum_aligned + Arm 4: hybrid_aligned -----------------
    if HAS_ALIGNMENT:
        try:
            t0 = time.time()
            aligned_kernel = train_aligned_kernel(Xq_train, y_train, Xq_train.shape[1])
            print(f"[progress] fold {fold_num}/{total_folds} kernel alignment: {time.time()-t0:.1f}s",
                  flush=True)

            t0 = time.time()
            qsvm_aligned = QSVC(quantum_kernel=aligned_kernel, probability=False)
            qsvm_aligned.fit(Xq_train, y_train)
            y_pred = np.asarray(qsvm_aligned.predict(Xq_test))
            rows.append({"fold": fold_num, "arm": "quantum_aligned",
                         **_metrics(y_test, y_pred, _proba_like(qsvm_aligned, Xq_test))})
            print(f"[progress] fold {fold_num}/{total_folds} quantum_aligned: {time.time()-t0:.1f}s "
                  f"(acc={rows[-1]['accuracy']:.3f})", flush=True)

            # hybrid_aligned: quantum-kernel-PCA on the ALIGNED kernel + classical features
            t0 = time.time()
            K_train = aligned_kernel.evaluate(Xq_train)
            K_test = aligned_kernel.evaluate(Xq_test, Xq_train)
            kpca = KernelPCA(n_components=N_HYBRID_QFEATURES, kernel="precomputed",
                              random_state=RANDOM_STATE)
            Zq_train = kpca.fit_transform(K_train)
            Zq_test = kpca.transform(K_test)

            qfeat_names = [f"qkpca_aligned_{i}" for i in range(Zq_train.shape[1])]
            hybrid_pool_names = classical_feats + qfeat_names
            Xh_train_full = np.hstack([X_train_imp[:, idx_c], Zq_train])
            Xh_test_full = np.hstack([X_test_imp[:, idx_c], Zq_test])

            hybrid_selected = select_features_lasso(
                Xh_train_full, y_train, hybrid_pool_names,
                min(N_FEATURES_FINAL_HYBRID, len(hybrid_pool_names)))
            idx_h = [hybrid_pool_names.index(f) for f in hybrid_selected]
            n_q_kept = sum(1 for f in hybrid_selected if f.startswith("qkpca_aligned_"))

            scaler_h = StandardScaler()
            Xh_train = scaler_h.fit_transform(Xh_train_full[:, idx_h])
            Xh_test = scaler_h.transform(Xh_test_full[:, idx_h])
            clf_h = clone(build_classical_model())
            clf_h.fit(Xh_train, y_train)
            rows.append({"fold": fold_num, "arm": "hybrid_aligned", "n_quantum_features_kept": n_q_kept,
                         **_metrics(y_test, clf_h.predict(Xh_test), _proba_like(clf_h, Xh_test))})
            print(f"[progress] fold {fold_num}/{total_folds} hybrid_aligned: {time.time()-t0:.1f}s "
                  f"(acc={rows[-1]['accuracy']:.3f}, kept {n_q_kept}/{len(qfeat_names)} quantum feats)",
                  flush=True)
        except Exception as e:
            print(f"[progress] fold {fold_num}/{total_folds} alignment pipeline FAILED "
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
        delayed(_run_one_fold)(i + 1, total_folds, tr, te, X_pool, y, pool_names, dataset_name, strategy_name)
        for i, (tr, te) in enumerate(splits)
    )
    fold_rows = [r for fold_rows in outcomes for r in fold_rows]
    print(f"[info] {dataset_name}/{strategy_name}: {total_folds} folds finished in "
          f"{(time.time()-run_start)/60:.1f} min", flush=True)
    return pd.DataFrame(fold_rows)


def paired_tests(fold_df, dataset_name, strategy_name):
    out = []
    piv_acc = fold_df.pivot_table(index="fold", columns="arm", values="accuracy")
    piv_f1 = fold_df.pivot_table(index="fold", columns="arm", values="f1")
    pairs = [("quantum_aligned", "quantum_fixed"),   # did alignment do anything at all?
             ("hybrid_aligned", "classical_only"),   # the actual target
             ("hybrid_aligned", "quantum_aligned")]
    for metric_name, piv in [("accuracy", piv_acc), ("f1", piv_f1)]:
        for a, b in pairs:
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
            out.append({"dataset": dataset_name, "feature_strategy": strategy_name, "metric": metric_name,
                        "arm_a": a, "arm_b": b, "n_folds_paired": len(paired),
                        "mean_a": paired[a].mean(), "mean_b": paired[b].mean(),
                        "mean_diff_a_minus_b": (paired[a] - paired[b]).mean(),
                        "ttest_p": t_p, "wilcoxon_p": w_p})
    return out


def main():
    if not HAS_QISKIT:
        print("[error] qiskit / qiskit-machine-learning not found. Install with:")
        print("    pip install qiskit qiskit-machine-learning qiskit-algorithms qiskit-aer")
        return
    if not HAS_ALIGNMENT:
        print(f"[warning] Kernel-alignment imports failed ({IMPORT_ERROR_MSG}). "
              "quantum_aligned/hybrid_aligned arms will be skipped; only classical_only and "
              "quantum_fixed will run. Send this error message back for a version-specific fix.")
    print(f"[info] qiskit-aer available: {HAS_AER}")

    csv_path = Path(CSV_PATH if len(sys.argv) < 2 else sys.argv[1])
    clinical_path = Path(CLINICAL_XLSX_PATH if len(sys.argv) < 3 else sys.argv[2])
    if not csv_path.exists() or not clinical_path.exists():
        print(f"[error] Check paths: {csv_path} / {clinical_path}")
        return

    out_dir = csv_path.parent / "model_outputs_kernel_alignment"
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

    all_fold_rows, all_tests = [], []
    for dataset_name, d in datasets.items():
        y = d["label_enc"].values
        X = d[feature_cols].values
        if min(np.bincount(y)) < N_SPLITS:
            print(f"[warning] {dataset_name}: minority class too small, skipping.")
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
    print("SUMMARY: classical_only vs quantum_fixed vs quantum_aligned vs hybrid_aligned")
    print("=" * 140)
    print(summary.to_string(index=False))

    if all_tests:
        tests_df = pd.DataFrame(all_tests)
        tests_df.to_csv(out_dir / "paired_tests.csv", index=False)
        print("\n" + "=" * 140)
        print("PAIRED TESTS  (quantum_aligned vs quantum_fixed = did alignment help AT ALL;")
        print(" hybrid_aligned vs classical_only = the actual target comparison)")
        print("=" * 140)
        print(tests_df.to_string(index=False))

    print(f"\n[done] All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
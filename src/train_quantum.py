"""
train_quantum_only.py
===================================
Quantum-only version of train_extended_boosted_1.py. All classical /
ensemble / boosting models (LDA, QDA, LogisticRegression, SVM, KNN,
RandomForest, XGBoost, LightGBM, etc.) have been removed. build_models()
now returns exactly three models, all from qiskit-machine-learning:

    QSVM  - Quantum Support Vector Machine (QSVC: a quantum kernel used
            inside a classical SVM)
    VQC   - Variational Quantum Classifier (a trained parametrized circuit)
    QNN   - Quantum Neural Network (EstimatorQNN + NeuralNetworkClassifier)

Requires qiskit, qiskit-machine-learning, and qiskit-algorithms to be
installed (qiskit-aer is optional but strongly recommended for speed) -
the script exits with an install hint at startup if they're missing.

Everything else is unchanged from the original pipeline: same subject-level
aggregation, same nested LASSO feature selection down to N_FEATURES_FINAL
features, same repeated-CV loop, same two dataset variants (A_Full /
B_StrictDysarthric). The 15 LASSO-selected features are further PCA-reduced
to N_QUBITS_QUANTUM components right before being encoded onto the quantum
circuit, since simulation cost grows steeply with qubit count.

Results are written to a NEW folder (model_outputs_quantum_only).

After the repeated-CV comparison finishes, it re-fits the single
best-performing (dataset, feature_strategy, model) combination on ALL
subjects in that dataset (no held-out fold - this is your final, deployable
model, not another CV estimate) and saves it as a .pkl file with joblib.

The .pkl bundles everything needed to run inference later:
    - "model"        the fitted classifier
    - "imputer"      the fitted SimpleImputer
    - "scaler"       the fitted StandardScaler
    - "selected_features"  which of the aggregated feature columns to select, in order
    - "label_encoder" maps 0/1 back to "ALS"/"HC"
    - "dataset_name", "feature_strategy", "model_name"  metadata for the report

To load and use it later:

    import joblib
    bundle = joblib.load("final_model.pkl")
    X_new_imputed = bundle["imputer"].transform(X_new[bundle["selected_features"]])
    X_new_scaled  = bundle["scaler"].transform(X_new_imputed)
    pred = bundle["model"].predict(X_new_scaled)
    label = bundle["label_encoder"].inverse_transform(pred)

Run:
    python train_quantum_only.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Quantum ML stack - required for this script (it's quantum-only). Install with:
#   pip install qiskit qiskit-machine-learning qiskit-algorithms
#   pip install qiskit-aer      # optional but strongly recommended, much faster
try:
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.algorithms import QSVC, VQC
    from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_algorithms.optimizers import COBYLA

    try:
        from qiskit_aer.primitives import SamplerV2 as AerSampler, EstimatorV2 as AerEstimator
        HAS_AER = True
    except ImportError:
        HAS_AER = False

    # ComputeUncompute is what FidelityQuantumKernel (used by QSVC) uses under
    # the hood to turn a sampler into a fidelity/kernel value. Without wiring
    # this to the Aer sampler explicitly, QSVC silently falls back to the slow
    # reference Sampler even when qiskit-aer is installed. Import path moved
    # between qiskit-machine-learning versions, so try both.
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
# CONFIG
# =============================================================================
CSV_PATH = r"D:\ABDM\Compressed\als_voice_features_with_literature_feats.csv"
CLINICAL_XLSX_PATH = r"D:\ABDM\Compressed\clinical_data.xlsx"

# N_REPEATS is deliberately much lower than a classical-only script would use
# (that original script used 20). Quantum circuit simulation is far slower
# than sklearn models: each fold trains all 3 quantum models, and VQC/QNN each
# run a QUANTUM_MAXITER-step optimizer loop on top of that. Raise this once
# you've confirmed the script runs end-to-end and know how long one repeat takes.
N_SPLITS = 5
# Dropped from 5 -> 3 repeats. This alone cuts total wall-clock time by ~40%
# with only a modest increase in metric variance. Bump back to 5 for your
# final confirmatory run once you're happy with the setup.
N_REPEATS = 3
RANDOM_STATE = 42

# N_JOBS: folds are independent, so they can run in parallel worker processes
# instead of one at a time. Defaults to (all cores - 1). CAUTION: qiskit-aer
# may already use multiple threads per simulation internally, so running many
# worker processes at once can cause CPU contention that eats into the gain -
# if you don't see close to a linear speedup, try a smaller N_JOBS (e.g. half
# your core count) and compare. Set to 1 to disable parallelism entirely.
N_JOBS = max(1, (os.cpu_count() or 4) - 1)
N_FEATURES_FINAL = 15
DYSARTHRIC_CUTOFF = 2
ID_COLS = ["filename", "filepath", "label", "subject_id", "task"]

# --- Fast-iteration toggles ------------------------------------------------
# Comment items out of these two lists to skip them entirely while you're
# iterating on speed/accuracy. Put everything back for your final run.
DATASETS_TO_RUN = ["A_Full_ALSvsHC", "B_StrictDysarthricALSvsHC"]
STRATEGIES_TO_RUN = ["Full_214_LASSO", "Curated_Pool_LASSO"]

# QSVM (the kernel method) is by FAR the slowest of the three models - its
# cost scales with train_size x test_size worth of circuit evaluations,
# which is why it alone took ~30+ min/fold on the full dataset in your log
# while VQC took ~2 min and QNN took ~10-15s. While you're iterating quickly,
# drop "QSVM" from this list; add it back for the run whose numbers you'll
# actually report/submit.
MODELS_TO_RUN = ["QSVM", "VQC", "QNN"]

# Number of shots used by the Aer sampler/estimator for the quantum circuits.
# Lower = faster but noisier expectation-value/fidelity estimates. 1024 is
# qiskit-aer's usual default; 512 roughly halves per-circuit sampling cost
# with only a small precision hit at this qubit count. If your installed
# qiskit-aer version doesn't accept this kwarg, the code below falls back to
# the library default automatically (see AerSampler/AerEstimator wiring).
SHOTS = 512

# Class-imbalance fix: caps the majority class in EACH TRAINING FOLD to at
# most this multiple of the minority class count (test folds are left
# untouched, so evaluation stays on the real, imbalanced distribution).
# This is why your QSVM numbers looked "good" but had ROC AUC ~0.50 on
# Dataset B (51 HC / 18 ALS) - the model was just learning to predict HC
# most of the time. Undersampling the majority class also *shrinks* the
# training set, which is a nice side effect: less data for QSVM's kernel to
# chew through means faster folds, not just fairer ones. Set to None to
# disable.
BALANCE_MAX_RATIO = 1.5

# --- Quantum model settings ---------------------------------------------
# N_QUBITS_QUANTUM: the 15 LASSO-selected features are further PCA-reduced
#   to this many components before being encoded onto the circuit, because
#   simulation cost grows steeply with qubit count. 4 is a reasonable
#   starting point on a laptop; try to keep this <= 6-8 without Aer/GPU.
# QUANTUM_REPS: circuit depth (repetitions) of the feature map / ansatz.
# QUANTUM_MAXITER: optimizer iterations for VQC / QNN training (QSVM doesn't
#   use this - it's a kernel method, not a trained variational circuit).
N_QUBITS_QUANTUM = 4
QUANTUM_REPS = 2
QUANTUM_MAXITER = 50
# QSVC's probability=True triggers sklearn's internal 5-fold Platt-scaling
# calibration - meaning it silently builds ~5 EXTRA quantum kernel matrices
# per fold just to get predict_proba. Set this False for a big speedup; you'll
# lose calibrated probabilities/roc_auc for QSVM specifically (predict/accuracy/
# f1 are unaffected - the wrapper's predict_proba fallback kicks in instead).
QSVC_PROBABILITY = False

# Set this to force which (dataset, feature_strategy, model) gets exported as
# the final .pkl. Leave as None to auto-pick the row with the highest f1_mean
# across the whole comparison table.
FORCE_EXPORT_CHOICE = None
# Example to force it:
# FORCE_EXPORT_CHOICE = {"dataset": "B_StrictDysarthricALSvsHC",
#                         "feature_strategy": "Full_214_LASSO",
#                         "model": "RandomForest"}

ESSENTIAL_RAW_FEATURES = [
    "PPE", "F0_std", "F0_min", "F0_CV",
    "HNR_mean", "HNR_std", "NHR", "GNE", "CPP",
    "Shimmer_APQ11", "Shimmer_DDA", "Shimmer_local_percent",
    "DFA", "RPDE", "Spread1", "Spread2",
    "PVI",
    "RelH2", "RelH7", "RelH8", "H2_std", "H7_std", "H8_mean",
    "MFCC_1_mean", "MFCC_8_mean",
]


# =============================================================================
# Clinical data loading (handles the Synapse placeholder-header quirk)
# =============================================================================
def load_clinical_data(xlsx_path: Path):
    sheets = pd.ExcelFile(xlsx_path).sheet_names
    sheet = "VOC-ALS_Data" if "VOC-ALS_Data" in sheets else 0
    probe = pd.read_excel(xlsx_path, sheet_name=sheet, header=None, nrows=2)
    row0_is_placeholder = probe.iloc[0].astype(str).str.match(r"^Column-\d+$").mean() > 0.8
    header_row = 1 if row0_is_placeholder else 0

    clin = pd.read_excel(xlsx_path, sheet_name=sheet, header=header_row)
    clin.columns = [str(c).strip() for c in clin.columns]
    if row0_is_placeholder:
        print(f"[info] Detected Synapse placeholder header row - using row {header_row} as the real header.")

    id_col = next((c for c in clin.columns if c.strip().lower() in ("id", "subject_id", "subjectid")), None)
    speech_col = next((c for c in clin.columns if "speech" in c.lower() and "subscore" in c.lower()), None)
    if id_col is None:
        raise ValueError(f"Couldn't find an ID column. Columns found: {list(clin.columns)}")

    clin = clin.rename(columns={id_col: "subject_id"})
    if speech_col:
        clin = clin.rename(columns={speech_col: "speech_subscore"})
        clin["speech_subscore"] = pd.to_numeric(clin["speech_subscore"], errors="coerce")
    return clin, speech_col is not None


# =============================================================================
# Subject-level aggregation
# =============================================================================
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


# =============================================================================
# Nested feature selection + model evaluation
# =============================================================================
def balance_undersample(X, y, max_ratio=1.5, random_state=42):
    """Cap the majority class at `max_ratio` x the minority class count.
    Only ever call this on a TRAINING split - never on test data, or you'd
    be evaluating on an artificial distribution. Cheaper than SMOTE/oversampling
    (shrinks rather than grows the training set), which is a bonus here since
    QSVM/VQC cost scales with training-set size. Returns (X, y) unchanged if
    max_ratio is None or the fold is already within ratio."""
    if max_ratio is None:
        return X, y
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)
    minority_count = counts.min()
    cap = max(int(minority_count * max_ratio), minority_count)
    keep_idx = []
    for cls, cnt in zip(classes, counts):
        idx = np.where(y == cls)[0]
        if cnt > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep_idx.append(idx)
    keep_idx = np.concatenate(keep_idx)
    rng.shuffle(keep_idx)
    return X[keep_idx], y[keep_idx]


def select_features_lasso(X_train, y_train, feature_names, n_final):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    lasso = LassoCV(cv=3, random_state=RANDOM_STATE, max_iter=10000)
    lasso.fit(X_scaled, y_train)
    importance = np.abs(lasso.coef_)
    order = np.argsort(importance)[::-1]
    return [feature_names[i] for i in order[:n_final]]


def _flatten(circuit):
    """Fully unroll a circuit's library/blueprint instructions (ZZFeatureMap,
    RealAmplitudes, etc.) down into basic gates. qiskit-aer's compiler can only
    execute basic gates - it doesn't recognize an opaque instruction just
    because it has a `.definition` - so anything built from qiskit's circuit
    library has to be decomposed before Aer touches it. Loops decompose()
    until the instruction set stops changing (one decompose() call often isn't
    enough - NLocal-based circuits like RealAmplitudes can be nested two levels
    deep), capped so a stray custom gate can't cause an infinite loop.
    """
    prev_ops = None
    for _ in range(6):
        ops = {instr.operation.name for instr in circuit.data}
        if ops == prev_ops:
            break
        prev_ops = ops
        circuit = circuit.decompose()
    return circuit


class QuantumClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Wraps a qiskit-machine-learning classifier (QSVC, VQC, or an EstimatorQNN-
    based QNN) so it behaves like any other sklearn classifier in this script:
    it supports fit / predict / predict_proba and sklearn.base.clone(), so it
    drops straight into run_repeated_cv()'s existing loop.

    kind="qsvc" -> Quantum Support Vector Machine (quantum kernel + classical SVM)
    kind="vqc"  -> Variational Quantum Classifier (trained variational circuit)
    kind="qnn"  -> Quantum Neural Network (EstimatorQNN + NeuralNetworkClassifier)

    Circuit simulation cost grows quickly with qubit count, so the classical
    features handed to this wrapper (already LASSO-selected upstream) are
    further PCA-reduced to `n_qubits` components, then min-max scaled to
    [0, pi] (a good input range for angle-encoding feature maps like ZZFeatureMap)
    before being encoded onto the circuit.
    """

    def __init__(self, kind="qsvc", n_qubits=4, reps=2, random_state=42, maxiter=100):
        self.kind = kind
        self.n_qubits = n_qubits
        self.reps = reps
        self.random_state = random_state
        self.maxiter = maxiter

    def _sampler(self):
        if not HAS_AER:
            return None
        try:
            return AerSampler(default_shots=SHOTS)
        except TypeError:
            # Older/newer qiskit-aer builds may not accept this kwarg name -
            # fall back to the library default rather than crashing the run.
            return AerSampler()

    def _estimator(self):
        if not HAS_AER:
            return None
        try:
            return AerEstimator(default_precision=None, options={"default_shots": SHOTS})
        except TypeError:
            return AerEstimator()

    def _build(self):
        # qiskit 2.x builds ZZFeatureMap/RealAmplitudes as opaque "blueprint"
        # instructions (a single gate literally named e.g. "ZZFeatureMap").
        # qiskit-aer's compiler doesn't know how to execute that gate directly -
        # it needs the circuit unrolled into basic gates (h, rz, cx, ...) first,
        # which is what _flatten() below does. Without this you'll hit
        # "AerError: unknown instruction: ZZFeatureMap" the moment Aer tries to
        # simulate anything.
        feature_map = _flatten(ZZFeatureMap(feature_dimension=self.n_qubits, reps=self.reps,
                                             entanglement="linear"))
        if self.kind == "qsvc":
            sampler = self._sampler()
            if sampler is not None:
                fidelity = ComputeUncompute(sampler=sampler)
                kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
            else:
                kernel = FidelityQuantumKernel(feature_map=feature_map)
            return QSVC(quantum_kernel=kernel, probability=QSVC_PROBABILITY)

        elif self.kind == "vqc":
            ansatz = _flatten(RealAmplitudes(num_qubits=self.n_qubits, reps=self.reps))
            optimizer = COBYLA(maxiter=self.maxiter)
            kwargs = {"sampler": self._sampler()} if HAS_AER else {}
            return VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer,
                       loss="cross_entropy", **kwargs)

        elif self.kind == "qnn":
            ansatz = _flatten(RealAmplitudes(num_qubits=self.n_qubits, reps=self.reps))
            circuit = feature_map.compose(ansatz)
            observable = SparsePauliOp("Z" + "I" * (self.n_qubits - 1))
            kwargs = {"estimator": self._estimator()} if HAS_AER else {}
            qnn = EstimatorQNN(circuit=circuit, observables=observable,
                                input_params=feature_map.parameters,
                                weight_params=ansatz.parameters, **kwargs)
            optimizer = COBYLA(maxiter=self.maxiter)
            return NeuralNetworkClassifier(neural_network=qnn, optimizer=optimizer,
                                            loss="squared_error", one_hot=False)
        else:
            raise ValueError(f"Unknown quantum model kind: {self.kind}")

    def _encode(self, X):
        X_reduced = self.pca_.transform(X)
        return self.qscaler_.transform(X_reduced)

    def fit(self, X, y):
        n_comp = min(self.n_qubits, X.shape[1], X.shape[0])
        self.pca_ = PCA(n_components=n_comp, random_state=self.random_state)
        X_reduced = self.pca_.fit_transform(X)
        self.qscaler_ = MinMaxScaler(feature_range=(0, np.pi))
        X_enc = self.qscaler_.fit_transform(X_reduced)
        self.classes_ = np.unique(y)
        self.n_qubits_used_ = n_comp
        self.model_ = self._build() if n_comp == self.n_qubits else self._build_for(n_comp)
        self.model_.fit(X_enc, y)
        return self

    def _build_for(self, n_comp):
        # Rare fallback: only hit if a CV fold has fewer samples/features than
        # n_qubits (e.g. a tiny minority class fold). Rebuild circuits at the
        # smaller size rather than crashing the fold.
        saved = self.n_qubits
        self.n_qubits = n_comp
        try:
            return self._build()
        finally:
            self.n_qubits = saved

    def _snap_to_classes(self, raw):
        # QSVC/VQC reliably predict() in {0,1} already. QNN (NeuralNetworkClassifier
        # wrapped around EstimatorQNN, a continuous-output regressor under the hood)
        # is not guaranteed to - depending on qiskit-machine-learning version it can
        # come back as {-1,+1} or other raw values, which sklearn's metrics then see
        # as a spurious 3rd class ("multiclass but average='binary'"). Snap whatever
        # comes back to the nearest actual label from fit(), for all three models,
        # so downstream metrics always see valid {0,1} labels.
        raw = np.asarray(raw, dtype=float).reshape(-1)
        classes = np.asarray(self.classes_, dtype=float)
        idx = np.argmin(np.abs(raw[:, None] - classes[None, :]), axis=1)
        return classes[idx].astype(self.classes_.dtype), idx

    def predict(self, X):
        labels, _ = self._snap_to_classes(self.model_.predict(self._encode(X)))
        return labels

    def predict_proba(self, X):
        X_enc = self._encode(X)
        try:
            proba = self.model_.predict_proba(X_enc)
            if proba is not None and proba.shape[1] == 2:
                return proba
        except Exception:
            pass
        # QSVC_PROBABILITY is off (Platt-scaling calibration is expensive), so
        # predict_proba above fails for QSVC - but SVC-based models still
        # expose decision_function, a continuous ranking score that's much
        # more informative for ROC AUC than collapsing straight to hard 0/1
        # predictions. This is what was making roc_auc_mean sit at ~0.50 even
        # when accuracy/F1 looked fine: the old fallback had no real ranking
        # signal to give roc_auc_score, only a single hard guess per sample.
        try:
            scores = np.asarray(self.model_.decision_function(X_enc), dtype=float).reshape(-1)
            lo, hi = scores.min(), scores.max()
            proba_pos = (scores - lo) / (hi - lo) if hi > lo else np.full_like(scores, 0.5)
            return np.column_stack([1 - proba_pos, proba_pos])
        except Exception:
            pass
        # Last resort (VQC/QNN, which have neither): fall back to hard
        # predictions as pseudo-probabilities so roc_auc degrades gracefully
        # instead of crashing the fold. Use the class-index mapping (not the
        # raw label value) so a QNN prediction of -1 doesn't wrap around to
        # column 1 via numpy's negative indexing.
        _, class_idx = self._snap_to_classes(self.model_.predict(X_enc))
        proba = np.zeros((len(class_idx), 2))
        proba[np.arange(len(class_idx)), class_idx] = 1.0
        return proba


def build_models():
    """Returns the three quantum models. Raises if qiskit isn't installed -
    call HAS_QISKIT-guarded code (see main()) before relying on this."""
    common = dict(n_qubits=N_QUBITS_QUANTUM, reps=QUANTUM_REPS, random_state=RANDOM_STATE)
    all_models = {
        "QSVM": QuantumClassifierWrapper(kind="qsvc", **common),
        "VQC": QuantumClassifierWrapper(kind="vqc", maxiter=QUANTUM_MAXITER, **common),
        "QNN": QuantumClassifierWrapper(kind="qnn", maxiter=QUANTUM_MAXITER, **common),
    }
    return {name: m for name, m in all_models.items() if name in MODELS_TO_RUN}


def _run_one_fold(fold_num, total_folds, train_idx, test_idx, X_pool, y, pool_names,
                   dataset_name, strategy_name, models):
    """Runs a single CV fold (feature selection + all 3 quantum models) and
    returns its results rather than mutating shared state, so this function
    can be dispatched to a separate worker process by joblib.Parallel."""
    fold_start = time.time()
    X_train, X_test = X_pool[train_idx], X_pool[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Balance the TRAINING split only - test_idx/y_test stay untouched so
    # metrics still reflect the real, imbalanced population.
    X_train, y_train = balance_undersample(X_train, y_train, BALANCE_MAX_RATIO, RANDOM_STATE)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    if min(np.bincount(y_train)) < 2:
        print(f"[progress] fold {fold_num}/{total_folds} ({dataset_name}/{strategy_name}) - "
              f"skipped (minority class too small in this split)", flush=True)
        return fold_num, {}, {}

    selected = select_features_lasso(X_train_imp, y_train, pool_names, N_FEATURES_FINAL)
    sel_idx = [pool_names.index(f) for f in selected]
    X_train_sel, X_test_sel = X_train_imp[:, sel_idx], X_test_imp[:, sel_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    print(f"[progress] fold {fold_num}/{total_folds} ({dataset_name}/{strategy_name}) - "
          f"starting (train={len(train_idx)}, test={len(test_idx)})...", flush=True)

    fold_metrics, fold_errs = {}, {}
    for name, model in models.items():
        model_start = time.time()
        try:
            model_clone = clone(model)
            model_clone.fit(X_train_scaled, y_train)
            y_pred = model_clone.predict(X_test_scaled)
            y_proba = (model_clone.predict_proba(X_test_scaled)[:, 1]
                       if hasattr(model_clone, "predict_proba") else None)
            fold_metrics[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None and
                            len(np.unique(y_test)) > 1 else np.nan,
            }
            print(f"[progress]   fold {fold_num} {name}: done in {time.time() - model_start:.1f}s "
                  f"(f1={fold_metrics[name]['f1']:.3f})", flush=True)
        except Exception as e:
            print(f"[progress]   fold {fold_num} {name}: FAILED after {time.time() - model_start:.1f}s "
                  f"({type(e).__name__})", flush=True)
            fold_errs[name] = (type(e).__name__, str(e), traceback.format_exc())
            continue

    print(f"[progress] fold {fold_num}/{total_folds} complete in {time.time() - fold_start:.1f}s", flush=True)
    return fold_num, fold_metrics, fold_errs


def run_repeated_cv(X, y, feature_names, feature_pool, strategy_name, dataset_name, out_dir=None):
    pool_idx = [feature_names.index(f) for f in feature_pool]
    X_pool = X[:, pool_idx]
    pool_names = feature_pool

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    models = build_models()
    splits = list(rskf.split(X_pool, y))
    total_folds = len(splits)

    fold_results = {name: [] for name in models}
    fold_errors = {name: None for name in models}  # first exception per model, for diagnostics

    run_start = time.time()
    print(f"[info] Running {total_folds} folds across up to {N_JOBS} parallel worker(s) "
          f"(set N_JOBS=1 near the top of the script to disable parallelism)...", flush=True)

    # Folds are independent, so hand them to a process pool instead of a
    # plain for-loop. Each worker builds/fits its own model clones, so no
    # fitted state is shared across processes.
    outcomes = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_run_one_fold)(i + 1, total_folds, train_idx, test_idx, X_pool, y,
                                pool_names, dataset_name, strategy_name, models)
        for i, (train_idx, test_idx) in enumerate(splits)
    )

    for fold_num, fold_metrics, fold_errs in outcomes:
        for name, metrics in fold_metrics.items():
            fold_results[name].append(metrics)
        for name, err in fold_errs.items():
            if fold_errors[name] is None:
                fold_errors[name] = err
                print(f"[warning] {name} failed on fold {fold_num} ({dataset_name}/{strategy_name}): "
                      f"{err[0]}: {err[1]}")
                print(err[2])  # traceback captured in the worker process

    elapsed = time.time() - run_start
    print(f"[info] {dataset_name}/{strategy_name}: {total_folds} folds finished in "
          f"{elapsed/60:.1f} min total ({elapsed/total_folds:.1f}s/fold average, wall-clock)", flush=True)

    for name, err in fold_errors.items():
        if err is not None and not fold_results[name]:
            print(f"[error] {name} produced ZERO successful folds in {dataset_name}/{strategy_name} - "
                  f"it will be missing from the results table. See the traceback above for why.")

    rows = []
    fold_detail_rows = []
    for name, results in fold_results.items():
        if not results:
            continue
        df_r = pd.DataFrame(results)
        for fold_i, metrics in enumerate(results, start=1):
            fold_detail_rows.append({"dataset": dataset_name, "feature_strategy": strategy_name,
                                      "model": name, "fold": fold_i, **metrics})
        rows.append({
            "dataset": dataset_name, "feature_strategy": strategy_name, "model": name,
            "n_folds": len(df_r),
            "accuracy_mean": df_r["accuracy"].mean(), "accuracy_std": df_r["accuracy"].std(),
            "f1_mean": df_r["f1"].mean(), "f1_std": df_r["f1"].std(),
            "precision_mean": df_r["precision"].mean(), "recall_mean": df_r["recall"].mean(),
            "roc_auc_mean": df_r["roc_auc"].mean(), "roc_auc_std": df_r["roc_auc"].std(),
        })

    # Checkpoint: append raw per-fold numbers as soon as this (dataset,
    # strategy) combo finishes, so a crash/interrupt later in a 4-hour run
    # doesn't lose everything - you can reload this CSV and keep going, or
    # just inspect it directly for fold-to-fold stability.
    if out_dir is not None and fold_detail_rows:
        detail_path = Path(out_dir) / "fold_level_results.csv"
        df_detail = pd.DataFrame(fold_detail_rows)
        df_detail.to_csv(detail_path, mode="a", header=not detail_path.exists(), index=False)
        print(f"[checkpoint] Appended {len(df_detail)} fold-level rows to {detail_path.name}", flush=True)

    return pd.DataFrame(rows)


# =============================================================================
# NEW: fit the final chosen config on ALL subjects and export as .pkl
# =============================================================================
def train_and_export_final_model(datasets, feature_cols, curated_pool, le, combined, out_dir, choice):
    dataset_name = choice["dataset"]
    strategy_name = choice["feature_strategy"]
    model_name = choice["model"]

    d = datasets[dataset_name]
    y = d["label_enc"].values
    X = d[feature_cols].values
    pool = feature_cols if strategy_name == "Full_214_LASSO" else curated_pool

    X_pool_all = X[:, [feature_cols.index(f) for f in pool]]
    X_pool_all, y = balance_undersample(X_pool_all, y, BALANCE_MAX_RATIO, RANDOM_STATE)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_pool_all)

    selected = select_features_lasso(X_imp, y, pool, N_FEATURES_FINAL)
    sel_idx = [pool.index(f) for f in selected]
    X_sel = X_imp[:, sel_idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)

    model = build_models()[model_name]
    model.fit(X_scaled, y)

    bundle = {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "pool_columns": pool,          # columns the imputer expects, in order
        "selected_features": selected, # subset (of pool) the model actually uses, in order
        "label_encoder": le,
        "dataset_name": dataset_name,
        "feature_strategy": strategy_name,
        "model_name": model_name,
        "n_train_subjects": len(d),
    }
    pkl_path = out_dir / "final_model.pkl"
    joblib.dump(bundle, pkl_path)
    print(f"\n[done] Final model exported: {pkl_path}")
    print(f"       Trained on {len(d)} subjects from '{dataset_name}', "
          f"{strategy_name}, {model_name}, {len(selected)} features.")
    return pkl_path


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not HAS_QISKIT:
        print("[error] qiskit / qiskit-machine-learning / qiskit-algorithms not found.")
        print("        Install with:")
        print("          pip install qiskit qiskit-machine-learning qiskit-algorithms")
        print("          pip install qiskit-aer   # optional, much faster simulator")
        return

    print(f"[info] qiskit-aer available: {HAS_AER}"
          + ("" if HAS_AER else "  <-- NOT installed! Quantum circuits will run on the "
                                 "slow reference simulator. Run: pip install qiskit-aer"))

    csv_path = Path(CSV_PATH if len(sys.argv) < 2 else sys.argv[1])
    clinical_path = Path(CLINICAL_XLSX_PATH if len(sys.argv) < 3 else sys.argv[2])

    if not csv_path.exists():
        print(f"[error] Features CSV not found: {csv_path}")
        return
    if not clinical_path.exists():
        print(f"[error] Clinical xlsx not found: {clinical_path}")
        return

    # Separate output folder so this quantum-only run never overwrites results
    # from any classical-model run.
    out_dir = csv_path.parent / "model_outputs_quantum_only"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Fresh checkpoint files for this run - avoids mixing rows from an older,
    # unrelated run into today's fold_level_results.csv / *_partial.csv.
    for stale in ("fold_level_results.csv", "robust_cv_results_partial.csv"):
        (out_dir / stale).unlink(missing_ok=True)

    df = pd.read_csv(csv_path)
    feature_cols_raw = [c for c in df.columns if c not in ID_COLS]
    df = df.dropna(subset=feature_cols_raw).reset_index(drop=True)

    clin, has_severity = load_clinical_data(clinical_path)
    matched = df[["subject_id"]].drop_duplicates().merge(clin[["subject_id"]], on="subject_id", how="left",
                                                           indicator=True)
    print(f"[info] Matched {(matched['_merge']=='both').sum()}/{len(matched)} subjects to clinical file.")

    subj_df, feature_cols = aggregate_to_subject_level(df)
    curated_pool = curated_pool_columns(feature_cols)
    print(f"[info] Curated candidate pool: {len(curated_pool)} columns "
          f"(from {len(ESSENTIAL_RAW_FEATURES)} literature/evidence-based raw features)")

    if has_severity:
        subj_df = subj_df.merge(clin[["subject_id", "speech_subscore"]], on="subject_id", how="left")

    datasets = {}
    le = LabelEncoder()
    subj_df["label_enc"] = le.fit_transform(subj_df["label"].values)
    datasets["A_Full_ALSvsHC"] = subj_df

    if has_severity:
        is_dysarthric = (subj_df["label"] == "ALS") & (subj_df["speech_subscore"] <= DYSARTHRIC_CUTOFF)
        is_hc = subj_df["label"] == "HC"
        subset = subj_df[is_dysarthric | is_hc].reset_index(drop=True)
        n_excluded = len(subj_df) - len(subset)
        print(f"[info] Strict dysarthric-only (subscore<={DYSARTHRIC_CUTOFF}): {len(subset)} subjects "
              f"({n_excluded} excluded)")
        datasets["B_StrictDysarthricALSvsHC"] = subset

    all_results = []
    for dataset_name, d in datasets.items():
        if dataset_name not in DATASETS_TO_RUN:
            print(f"[info] Skipping {dataset_name} (not in DATASETS_TO_RUN).")
            continue
        y = d["label_enc"].values
        X = d[feature_cols].values
        if min(np.bincount(y)) < N_SPLITS:
            print(f"[warning] {dataset_name}: minority class too small ({min(np.bincount(y))}) "
                  f"for {N_SPLITS}-fold CV. Skipping.")
            continue
        print(f"\n[info] === {dataset_name}: {len(d)} subjects "
              f"({np.bincount(y)[0]} {le.classes_[0]}, {np.bincount(y)[1]} {le.classes_[1]}) ===")

        for strategy_name, pool in [("Full_214_LASSO", feature_cols),
                                     ("Curated_Pool_LASSO", curated_pool)]:
            if strategy_name not in STRATEGIES_TO_RUN:
                print(f"[info] Skipping {strategy_name} (not in STRATEGIES_TO_RUN).")
                continue
            print(f"[info] Running repeated CV: {strategy_name} ({len(pool)} candidates), "
                  f"quantum models (PCA-reduced to {N_QUBITS_QUANTUM} qubits), "
                  f"models={MODELS_TO_RUN}...", flush=True)
            res = run_repeated_cv(X, y, feature_cols, pool, strategy_name, dataset_name, out_dir=out_dir)
            all_results.append(res)

            # Checkpoint the running summary too, not just the raw fold rows -
            # so if the process dies mid-run you still have a readable
            # results table for everything completed so far.
            partial = pd.concat(all_results, ignore_index=True)
            partial.to_csv(out_dir / "robust_cv_results_partial.csv", index=False)
            print(f"[checkpoint] Saved {len(partial)}-row partial summary to "
                  f"robust_cv_results_partial.csv", flush=True)

    if not all_results:
        print("[error] No results produced.")
        return
    combined = pd.concat(all_results, ignore_index=True)
    if combined.empty or "dataset" not in combined.columns:
        print("\n[error] Every quantum model failed on every fold, for every dataset/strategy - "
              "the results table is empty. Scroll up to the '[warning] ... failed on a fold' "
              "messages above for the actual exception and traceback from the first failure of "
              "each model. Common causes: an incompatible qiskit/qiskit-machine-learning/"
              "qiskit-algorithms version combo, or a qiskit-aer primitive signature mismatch.")
        return
    combined = combined.sort_values(["dataset", "f1_mean"], ascending=[True, False])
    combined.to_csv(out_dir / "robust_cv_results.csv", index=False)

    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}" if isinstance(v, float) else str(v))
    print("\n" + "=" * 130)
    print(f"REPEATED CV RESULTS ({N_SPLITS}-fold x {N_REPEATS} repeats = up to {N_SPLITS*N_REPEATS} folds each)")
    print("=" * 130)
    print(combined.to_string(index=False))

    fig, ax = plt.subplots(figsize=(13, 7))
    x_labels = [f"{r.dataset}\n{r.feature_strategy}\n{r.model}" for r in combined.itertuples()]
    x = np.arange(len(combined))
    ax.bar(x, combined["f1_mean"], yerr=combined["f1_std"], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
    ax.set_ylabel("Test F1 (mean +/- std across folds)")
    ax.set_title("Repeated CV: dataset x feature strategy x model")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "robust_cv_comparison.png", dpi=150)
    plt.close(fig)

    # ---- NEW: export the best (or forced) config as a final .pkl model ----
    if FORCE_EXPORT_CHOICE is not None:
        choice = FORCE_EXPORT_CHOICE
    else:
        best_row = combined.sort_values("f1_mean", ascending=False).iloc[0]
        choice = {"dataset": best_row["dataset"], "feature_strategy": best_row["feature_strategy"],
                  "model": best_row["model"]}
    train_and_export_final_model(datasets, feature_cols, curated_pool, le, combined, out_dir, choice)

    print(f"\n[done] Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()

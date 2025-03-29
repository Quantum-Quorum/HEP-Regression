import os
import traceback

import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
# Consider RandomForestClassifier if treating n_jets separately
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.metrics import log_loss, accuracy_score # For classification metrics if needed
# from scipy.stats import entropy # For KL Divergence calculation later
import pennylane as qml
from tqdm import tqdm
import io
from sklearn.multioutput import MultiOutputRegressor
import time  # To time operations


class LorentzVector:
    def __init__(self, px, py, pz, E):
        self.px = px
        self.py = py
        self.pz = pz
        self.E = E
        self._epsilon = 1e-10

    def pt(self):
        pt_sq = self.px ** 2 + self.py ** 2
        return np.sqrt(np.maximum(0, pt_sq))

    def eta(self):
        p_sq = self.px ** 2 + self.py ** 2 + self.pz ** 2
        p = np.sqrt(np.maximum(0, p_sq))
        p_plus_pz = p + self.pz
        p_minus_pz = p - self.pz

        if isinstance(p, np.ndarray):
            eta_vals = np.zeros_like(p, dtype=float)
            safe_mask = (p_minus_pz > self._epsilon) & (p_plus_pz > self._epsilon) & (p > self._epsilon)
            eta_vals[safe_mask] = 0.5 * np.log(p_plus_pz[safe_mask] / p_minus_pz[safe_mask])
            inf_mask_pos = (~safe_mask) & (p > self._epsilon) & (self.pz > 0)
            inf_mask_neg = (~safe_mask) & (p > self._epsilon) & (self.pz < 0)
            eta_vals[inf_mask_pos] = np.inf
            eta_vals[inf_mask_neg] = -np.inf
            return eta_vals
        else:
            if p <= self._epsilon: return 0.0
            if p_minus_pz <= self._epsilon: return np.inf
            if p_plus_pz <= self._epsilon: return -np.inf
            return 0.5 * np.log(p_plus_pz / p_minus_pz)

    def phi(self):
        return np.arctan2(self.py, self.px)

    def mass(self):
        mass_sq = self.E ** 2 - self.px ** 2 - self.py ** 2 - self.pz ** 2
        return np.sqrt(np.maximum(0.0, mass_sq))

    def __add__(self, other):
        if isinstance(other, LorentzVector):
            return LorentzVector(self.px + other.px,
                                 self.py + other.py,
                                 self.pz + other.pz,
                                 self.E + other.E)
        return NotImplemented


def compute_delta_r(eta1, phi1, eta2, phi2):
    delta_eta = eta1 - eta2
    delta_phi = phi1 - phi2
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    if not (np.all(np.isfinite(delta_eta)) and np.all(np.isfinite(delta_phi))):
        return 10.0  # Arbitrary large value, might need refinement
    return np.sqrt(delta_eta ** 2 + delta_phi ** 2)


def extract_parton_features(partons_raw_df):
    """Computes features solely from the input parton data."""
    features = []
    required_cols = [f'parton_{i}_{comp}' for i in range(2) for comp in ['px', 'py', 'pz', 'E', 'id', 'charge']]
    if not all(col in partons_raw_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in partons_raw_df.columns]
        raise ValueError(f"Missing required raw parton columns: {missing}")

    print("Extracting Parton Features...")
    for index, row in tqdm(partons_raw_df.iterrows(), total=partons_raw_df.shape[0]):
        try:
            p0_vec = LorentzVector(px=row['parton_0_px'], py=row['parton_0_py'], pz=row['parton_0_pz'],
                                   E=row['parton_0_E'])
            p1_vec = LorentzVector(px=row['parton_1_px'], py=row['parton_1_py'], pz=row['parton_1_pz'],
                                   E=row['parton_1_E'])
            p_system = p0_vec + p1_vec

            p0_pt, p0_eta, p0_phi, p0_mass = p0_vec.pt(), p0_vec.eta(), p0_vec.phi(), p0_vec.mass()
            p1_pt, p1_eta, p1_phi, p1_mass = p1_vec.pt(), p1_vec.eta(), p1_vec.phi(), p1_vec.mass()

            if not (np.isfinite(p0_eta) and np.isfinite(p1_eta)):
                parton_delta_eta = np.nan
                parton_delta_phi_wrap = np.nan
                parton_delta_R = np.nan
            else:
                parton_delta_eta = p0_eta - p1_eta
                dphi = p0_phi - p1_phi
                parton_delta_phi_wrap = (dphi + np.pi) % (2 * np.pi) - np.pi
                parton_delta_R = np.sqrt(parton_delta_eta ** 2 + parton_delta_phi_wrap ** 2)

            event_features = {
                'event_id': index,
                'p0_pt': p0_pt, 'p0_eta': p0_eta, 'p0_phi': p0_phi, 'p0_mass': p0_mass,
                'p0_id': row['parton_0_id'], 'p0_charge': row['parton_0_charge'],
                'p1_pt': p1_pt, 'p1_eta': p1_eta, 'p1_phi': p1_phi, 'p1_mass': p1_mass,
                'p1_id': row['parton_1_id'], 'p1_charge': row['parton_1_charge'],
                'parton_delta_eta': parton_delta_eta,
                'parton_delta_phi_abs': np.abs(parton_delta_phi_wrap),  # Use abs value? or keep sign?
                'parton_delta_phi_wrap': parton_delta_phi_wrap,  # Keep wrapped version
                'parton_delta_R': parton_delta_R,
                'parton_system_pt': p_system.pt(),
                'parton_system_eta': p_system.eta(),  # Might also be inf
                'parton_system_phi': p_system.phi(),
                'parton_system_mass': p_system.mass(),
                'parton_dot_prod': p0_vec.E * p1_vec.E - p0_vec.px * p1_vec.px - p0_vec.py * p1_vec.py - p0_vec.pz * p1_vec.pz,
            }
            features.append(event_features)
        except Exception as e:
            print(f"Warning: Error processing partons in event {index}: {e}")
            continue

    feature_df = pd.DataFrame(features)
    large_finite_val = 1e5
    feature_df.replace([np.inf, -np.inf], [large_finite_val, -large_finite_val], inplace=True)
    initial_count = len(feature_df)
    feature_df.dropna(inplace=True)
    if len(feature_df) < initial_count:
        print(f"Dropped {initial_count - len(feature_df)} rows containing NaNs during feature extraction.")

    if feature_df.empty:
        print("Warning: Feature DataFrame is empty after processing and NaN handling.")
    else:
        feature_df['event_id'] = feature_df['event_id'].astype(int)
    return feature_df


# --- Function to Extract Jet Targets (y) ---
def extract_jet_targets(jets_raw_df, num_max_jets):
    """Calculates target variables (n_jets, leading_pt, subleading_pt) from raw jet data."""
    targets = []
    print("Extracting Jet Targets...")
    for index, row in tqdm(jets_raw_df.iterrows(), total=jets_raw_df.shape[0]):
        valid_jets = []
        for j in range(num_max_jets):
            px_val = row.get(f'jet_{j}_px', 0)
            py_val = row.get(f'jet_{j}_py', 0)
            pz_val = row.get(f'jet_{j}_pz', 0)
            E_val = row.get(f'jet_{j}_E', 0)

            if px_val == 0 and py_val == 0 and pz_val == 0 and E_val == 0:
                continue

            try:
                pt_val = np.sqrt(px_val ** 2 + py_val ** 2)
                if pt_val > 1e-6 and np.isfinite(pt_val):  # Use small threshold > 0
                    valid_jets.append({'pt': pt_val})
            except Exception as e:
                print(f"Warning: Error calculating pt for jet {j} in event {index}: {e}")
                continue

        valid_jets.sort(key=lambda x: x['pt'], reverse=True)

        n_jets = len(valid_jets)
        leading_pt = valid_jets[0]['pt'] if n_jets > 0 else 0.0
        subleading_pt = valid_jets[1]['pt'] if n_jets > 1 else 0.0

        targets.append({
            'event_id': index,
            'n_jets': n_jets,
            'leading_pt': leading_pt,
            'subleading_pt': subleading_pt
        })

    target_df = pd.DataFrame(targets)
    if target_df.empty:
        print("Warning: Target DataFrame is empty after processing.")
    else:
        target_df['event_id'] = target_df['event_id'].astype(int)
    return target_df


# --- Modified Data Processing Function ---
def process_training_data(file_path):
    """Loads parton and jet data, extracts features (X) from partons, extracts targets (y) from jets, and merges them."""
    parton_features_df = None
    jet_targets_df = None
    merged_df = None

    start_time = time.time()
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\n--- Processing training data from: {file_path} ---")
            keys = list(f.keys())
            print(f'Keys found: {keys}')

            # --- Process Partons for Features ---
            if 'partons' in keys:
                partons_data = f['partons'][:]
                print(f"Raw parton data shape: {partons_data.shape}")
                if partons_data.ndim == 3 and partons_data.shape[1] == 2 and partons_data.shape[2] == 6:
                    num_events = partons_data.shape[0]
                    column_names = [f'parton_{i}_{comp}' for i in range(2) for comp in
                                    ['px', 'py', 'pz', 'E', 'id', 'charge']]
                    partons_raw_df = pd.DataFrame(partons_data.reshape(num_events, -1), columns=column_names)
                    parton_features_df = extract_parton_features(partons_raw_df)
                else:
                    print(f"Error: Unexpected parton data shape: {partons_data.shape}. Expected (N, 2, 6).")
            else:
                print("Error: 'partons' dataset not found. Cannot extract features.")
                return None

            # --- Process Jets for Targets ---
            if 'jets' in keys:
                jets_data = f['jets'][:]
                print(f"Raw jet data shape: {jets_data.shape}")
                if jets_data.ndim == 3 and jets_data.shape[2] == 4:  # N_event x N_max_jets x 4 (px,py,pz,E)
                    num_events_jets = jets_data.shape[0]
                    num_max_jets = jets_data.shape[1]
                    jet_column_names = [f'jet_{j}_{comp}' for j in range(num_max_jets) for comp in
                                        ['px', 'py', 'pz', 'E']]
                    jets_raw_df = pd.DataFrame(jets_data.reshape(num_events_jets, -1), columns=jet_column_names)
                    jet_targets_df = extract_jet_targets(jets_raw_df, num_max_jets)
                else:
                    print(f"Error: Unexpected jet data shape: {jets_data.shape}. Expected (N, N_max, 4).")
            else:
                print("Error: 'jets' dataset not found. Cannot extract targets.")
                return None

            # --- Merge Features and Targets ---
            if parton_features_df is not None and not parton_features_df.empty and \
                    jet_targets_df is not None and not jet_targets_df.empty:
                print(
                    f"Merging features ({len(parton_features_df)} events) and targets ({len(jet_targets_df)} events)...")
                # Use inner merge to keep only events present in both post-processing
                merged_df = pd.merge(parton_features_df, jet_targets_df, on='event_id', how='inner')
                print(f"Merged data shape: {merged_df.shape}")
                if merged_df.empty:
                    print("Error: Merged DataFrame is empty. Check processing steps and event IDs.")
                    return None
                # merged_df.set_index('event_id', inplace=True, drop=False) # Keep event_id as column too
            else:
                print("Error: Cannot merge due to empty features or targets DataFrame.")
                return None

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error processing training data file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

    end_time = time.time()
    print(f"Data processing finished in {end_time - start_time:.2f} seconds.")
    return merged_df


# --- Prepare Data for Regression (Returns scaled data and scalers) ---
def prepare_data_regression(feature_target_df, feature_columns, target_columns):
    """Prepares data, scales features and targets, returns scaled data and scalers."""
    print("\n--- Preparing data for regression ---")
    if feature_target_df is None or feature_target_df.empty:
        print("Input DataFrame is None or empty.")
        return None, None, None, None

    missing_features = [col for col in feature_columns if col not in feature_target_df.columns]
    missing_targets = [col for col in target_columns if col not in feature_target_df.columns]
    if missing_features or missing_targets:
        print(f"Error: Missing feature columns: {missing_features}")
        print(f"Error: Missing target columns: {missing_targets}")
        return None, None, None, None

    X = feature_target_df[feature_columns].values
    y = feature_target_df[target_columns].values
    print(f"Raw data shapes: X={X.shape}, y={y.shape}")

    if np.any(~np.isfinite(X)):
        print("Warning: Non-finite values found in feature data before scaling. Replacing with 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(~np.isfinite(y)):
        print("Warning: Non-finite values found in target data before scaling. Replacing with 0.")
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    scaler_x = StandardScaler()
    try:
        X_scaled = scaler_x.fit_transform(X)
    except ValueError as e:
        print(f"Error during feature scaling: {e}. Check for constant columns.")
        variances = np.var(X, axis=0)
        constant_cols_indices = np.where(variances < 1e-9)[0]
        if len(constant_cols_indices) > 0:
            constant_cols_names = [feature_columns[i] for i in constant_cols_indices]
            print(f"Detected constant feature columns: {constant_cols_names}. Consider removing them.")
        return None, None, None, None

    scaler_y = StandardScaler()
    try:
        y_scaled = scaler_y.fit_transform(y)
    except ValueError as e:
        print(f"Error during target scaling: {e}. Check for constant columns.")
        return None, None, None, None

    print(f"Scaling complete: X_scaled shape {X_scaled.shape}, y_scaled shape {y_scaled.shape}")
    return X_scaled, y_scaled, scaler_x, scaler_y


# --- Quantum Model Function ---
def main_regression_quantum_pennylane_fixed(X_train, X_test, y_train, y_test, feature_dimension,
                                            use_amplitude_embedding=False):
    """Runs Quantum Kernel Regression using PennyLane."""
    print(f"\n--- Running Quantum Kernel Regression ---")
    start_q_time = time.time()

    if use_amplitude_embedding:
        if feature_dimension <= 0:
            print("Error: Feature dimension must be positive for Amplitude Embedding.")
            return None, np.nan, np.nan
        n_qubits = int(np.ceil(np.log2(feature_dimension)))
        print(f"Using {n_qubits} qubits for Amplitude Embedding (feature dim {feature_dimension}).")
    else:  # Angle Embedding
        n_qubits = feature_dimension
        print(f"Using {n_qubits} qubits for Angle Embedding.")

    if n_qubits <= 0:
        print("Error: Number of qubits must be positive.")
        return None, np.nan, np.nan
    max_qksvr_qubits = 14
    if n_qubits > max_qksvr_qubits:
        print(
            f"Error: Requested qubits ({n_qubits}) exceeds practical limit for QK-SVR simulation ({max_qksvr_qubits}). Aborting.")
        return None, np.nan, np.nan

    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
        print("Using lightning.qubit device.")
    except qml.DeviceError:
        print("lightning.qubit not available, falling back to default.qubit.")
        dev = qml.device("default.qubit", wires=n_qubits)

    def angle_embedding_circuit(features):
        qml.broadcast(qml.Hadamard, wires=range(n_qubits), pattern="single")
        qml.broadcast(qml.RZ, wires=range(n_qubits), pattern="single", parameters=features)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

    def amplitude_embedding_circuit(features):
        target_dim = 1 << n_qubits
        if len(features) > target_dim:
            raise ValueError(
                f"Amplitude embedding requires feature length ({len(features)}) <= 2^n_qubits ({target_dim})")
        features_padded = np.pad(features, (0, target_dim - len(features)), 'constant') if len(
            features) < target_dim else features
        norm = np.linalg.norm(features_padded)
        state_vector = np.zeros(target_dim)
        if norm > 1e-10:
            state_vector = features_padded / norm
        else:
            state_vector[0] = 1.0
        qml.StatePrep(state_vector, wires=range(n_qubits))
        # qml.broadcast(qml.RY, wires=range(n_qubits), pattern='single', parameters=np.random.rand(n_qubits)*0.1) # Small random rotation

    # Quantum Kernel Circuit
    @qml.qnode(dev, interface="autograd")
    def kernel_circuit(features):
        if use_amplitude_embedding:
            amplitude_embedding_circuit(features)
        else:
            angle_embedding_circuit(features)
        return qml.state()

    print("Initializing Quantum Kernel...")
    kernel = qml.kernels.Kernel(kernel_circuit, wires=range(n_qubits))

    print("Computing Quantum Kernel Matrix for training data...")
    kernel_comp_start = time.time()
    quantum_kernel_train = kernel.matrix(X_train, X_train)
    kernel_comp_end = time.time()
    print(f"Training Kernel Matrix computed in {kernel_comp_end - kernel_comp_start:.2f} seconds.")

    # Train SVR
    print("Training Quantum Kernel SVR with Hyperparameter Tuning...")
    param_grid = {'C': [1, 10], 'epsilon': [0.1, 0.2]}
    svr_base = SVR(kernel='precomputed', cache_size=500)  # Increase cache size
    grid_search = GridSearchCV(svr_base, param_grid, cv=2, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')
    multi_output_qsvr = MultiOutputRegressor(grid_search, n_jobs=-1)  # Parallelize over outputs too

    train_start = time.time()
    multi_output_qsvr.fit(quantum_kernel_train, y_train)
    train_end = time.time()
    print(f"SVR Training finished in {train_end - train_start:.2f} seconds.")

    # Evaluate on Test Set
    print("Computing Quantum Kernel Matrix for test data...")
    kernel_test_start = time.time()
    quantum_kernel_test = kernel.matrix(X_test, X_train)
    kernel_test_end = time.time()
    print(f"Test Kernel Matrix computed in {kernel_test_end - kernel_test_start:.2f} seconds.")

    print("Evaluating Quantum Kernel SVR...")
    predict_start = time.time()
    y_pred_quantum_scaled = multi_output_qsvr.predict(quantum_kernel_test)
    predict_end = time.time()
    print(f"Prediction finished in {predict_end - predict_start:.2f} seconds.")

    # Metrics on SCALED data
    mse_quantum = mean_squared_error(y_test, y_pred_quantum_scaled)
    r2_quantum = r2_score(y_test, y_pred_quantum_scaled, multioutput='uniform_average')
    mse_per_output = mean_squared_error(y_test, y_pred_quantum_scaled, multioutput='raw_values')
    r2_per_output = r2_score(y_test, y_pred_quantum_scaled, multioutput='raw_values')

    end_q_time = time.time()
    print(f'\nQuantum Kernel SVR (PennyLane) Results (Total time: {end_q_time - start_q_time:.2f} s):')
    print(f'  - Embedding: {"Amplitude" if use_amplitude_embedding else "Angle"} ({n_qubits} qubits)')
    try:
        # Access best params for the first estimator; may fail if grid search failed
        print(f"  - Best SVR Params (1st target): {multi_output_qsvr.estimators_[0].best_params_}")
    except Exception:
        print("  - Could not retrieve best SVR params.")
    print(f'  - Overall MSE (scaled): {mse_quantum:.4f}')
    print(f'  - Overall R2 (scaled, uniform avg): {r2_quantum:.4f}')
    print(f'  - MSE per output (nJ, lPt, slPt) (scaled): {mse_per_output}')
    print(f'  - R2 per output (nJ, lPt, slPt) (scaled): {r2_per_output}')

    return y_pred_quantum_scaled, mse_quantum, r2_quantum


# --- Classical Model Function ---
def main_regression_classical(X_train, X_test, y_train, y_test):
    """Trains and evaluates a classical RandomForestRegressor, including tuning."""
    print(f"\n--- Running Classical Regression (RandomForest) ---")
    start_c_time = time.time()

    # --- Default RF ---
    print("Training default RandomForestRegressor...")
    model_default = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
    model_default.fit(X_train, y_train)
    print(f"Default RF OOB Score: {model_default.oob_score_ if hasattr(model_default, 'oob_score_') else 'N/A'}")

    print("Evaluating default RandomForestRegressor...")
    y_pred_default_scaled = model_default.predict(X_test)
    mse_default = mean_squared_error(y_test, y_pred_default_scaled)
    r2_default = r2_score(y_test, y_pred_default_scaled, multioutput='uniform_average')
    mse_default_per_output = mean_squared_error(y_test, y_pred_default_scaled, multioutput='raw_values')
    r2_default_per_output = r2_score(y_test, y_pred_default_scaled, multioutput='raw_values')

    print(f'\nDefault RandomForest Results:')
    print(f'  - Overall MSE (scaled): {mse_default:.4f}')
    print(f'  - Overall R2 (scaled, uniform avg): {r2_default:.4f}')
    print(f'  - MSE per output (nJ, lPt, slPt) (scaled): {mse_default_per_output}')
    print(f'  - R2 per output (nJ, lPt, slPt) (scaled): {r2_default_per_output}')

    # --- Hyperparameter Tuning ---
    print("\nRunning Hyperparameter Tuning for RandomForestRegressor...")
    # Expanded grid
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5],
        'max_features': ['sqrt', 0.5, 1.0]
    }
    rf_tune = RandomForestRegressor(random_state=42, n_jobs=-1)
    random_search = RandomizedSearchCV(estimator=rf_tune,
                                       param_distributions=param_grid_rf,
                                       n_iter=20,  # Increase iterations
                                       cv=3,
                                       verbose=1,
                                       n_jobs=-1,  # Parallelize CV folds
                                       random_state=42,
                                       scoring='neg_mean_squared_error')  # Optimize for MSE

    tune_start = time.time()
    random_search.fit(X_train, y_train)
    tune_end = time.time()
    print(f"Tuning finished in {tune_end - tune_start:.2f} seconds.")

    best_params = random_search.best_params_
    best_score = random_search.best_score_
    print(f"\nBest parameters found (RandomSearch): {best_params}")
    print(f"Best CV score (Negative MSE): {best_score:.4f}")

    best_rf_model = random_search.best_estimator_
    y_pred_tuned_scaled = best_rf_model.predict(X_test)
    mse_tuned = mean_squared_error(y_test, y_pred_tuned_scaled)
    r2_tuned = r2_score(y_test, y_pred_tuned_scaled, multioutput='uniform_average')
    mse_tuned_per_output = mean_squared_error(y_test, y_pred_tuned_scaled, multioutput='raw_values')
    r2_tuned_per_output = r2_score(y_test, y_pred_tuned_scaled, multioutput='raw_values')

    end_c_time = time.time()
    print(f'\nTuned Classical RandomForest Results (Total time: {end_c_time - start_c_time:.2f} s):')
    print(f'  - Test Overall MSE (scaled): {mse_tuned:.4f}')
    print(f'  - Test Overall R2 (scaled, uniform avg): {r2_tuned:.4f}')
    print(f'  - Test MSE per output (nJ, lPt, slPt) (scaled): {mse_tuned_per_output}')
    print(f'  - Test R2 per output (nJ, lPt, slPt) (scaled): {r2_tuned_per_output}')

    return best_rf_model, y_pred_tuned_scaled


def calculate_kl_divergence(p_counts, q_counts):
    """ Calculates KL divergence D_KL(P || Q) for two count distributions. """
    # Ensure inputs are numpy arrays
    p = np.asarray(p_counts, dtype=float)
    q = np.asarray(q_counts, dtype=float)

    p /= p.sum()
    q /= q.sum()

    epsilon = 1e-10
    p = np.where(p == 0, epsilon, p)
    q = np.where(q == 0, epsilon, q)

    return np.sum(p * np.log(p / q))


def evaluate_kl_divergence(y_true_orig, y_pred_orig, n_bins=50, pt_range=(0, 200)):
    """Calculates KL divergence for n_jets, leading_pt, subleading_pt distributions."""
    print("\n--- Evaluating KL Divergence ---")
    kl_scores = {}

    # Columns: 0=n_jets, 1=leading_pt, 2=subleading_pt

    true_njets = y_true_orig[:, 0].astype(int)
    pred_njets = np.round(y_pred_orig[:, 0]).astype(int)  # Round predictions
    pred_njets[pred_njets < 0] = 0  # Ensure non-negative

    max_njets = max(true_njets.max(), pred_njets.max())
    true_njets_counts = np.bincount(true_njets, minlength=max_njets + 1)
    pred_njets_counts = np.bincount(pred_njets, minlength=max_njets + 1)

    kl_njets = calculate_kl_divergence(true_njets_counts, pred_njets_counts)
    kl_scores['n_jets'] = kl_njets
    print(f"KL Divergence (n_jets): {kl_njets:.4f}")

    true_lpt = y_true_orig[:, 1]
    pred_lpt = y_pred_orig[:, 1]
    valid_lpt_mask = true_lpt > 1e-6

    true_lpt_hist, _ = np.histogram(true_lpt[valid_lpt_mask], bins=n_bins, range=pt_range)
    pred_lpt_hist, _ = np.histogram(pred_lpt[valid_lpt_mask], bins=n_bins, range=pt_range)  # Use same mask

    kl_lpt = calculate_kl_divergence(true_lpt_hist, pred_lpt_hist)
    kl_scores['leading_pt'] = kl_lpt
    print(f"KL Divergence (leading_pt): {kl_lpt:.4f}")

    true_slpt = y_true_orig[:, 2]
    pred_slpt = y_pred_orig[:, 2]
    valid_slpt_mask = true_slpt > 1e-6

    true_slpt_hist, _ = np.histogram(true_slpt[valid_slpt_mask], bins=n_bins, range=pt_range)
    pred_slpt_hist, _ = np.histogram(pred_slpt[valid_slpt_mask], bins=n_bins, range=pt_range)  # Use same mask

    kl_slpt = calculate_kl_divergence(true_slpt_hist, pred_slpt_hist)
    kl_scores['subleading_pt'] = kl_slpt
    print(f"KL Divergence (subleading_pt): {kl_slpt:.4f}")

    final_score = np.mean(list(kl_scores.values()))
    print(f"Average KL Divergence Score: {final_score:.4f}")

    return kl_scores, final_score


# --- Main Execution Block ---
if __name__ == "__main__":
    main_start_time = time.time()

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))
    print(repo_root)
    data_path = os.path.join(repo_root, 'data')
    output_path = os.path.join(repo_root, 'outputs')
    os.makedirs(output_path, exist_ok=True)
    print(f"Script directory: {script_dir}")
    print(f"Data directory: {data_path}")
    print(f"Output directory: {output_path}")

    training_data_fp = os.path.join(data_path, 'pp-z-to-jets-500K-57246.h5')
    if not os.path.exists(training_data_fp):
        print(f"FATAL ERROR: Training data file not found at {training_data_fp}")
        exit()

    # Process Training Data
    feature_target_df = process_training_data(training_data_fp)

    if feature_target_df is not None and not feature_target_df.empty:
        print(f"\nSuccessfully processed and merged data. Shape: {feature_target_df.shape}")
        # print(f"Columns: {feature_target_df.columns.tolist()}") # Optional: Print all columns

        feature_columns = [col for col in feature_target_df.columns if
                           col.startswith('p0_') or col.startswith('p1_') or col.startswith('parton_')]
        target_columns = ['n_jets', 'leading_pt', 'subleading_pt']

        if not all(col in feature_target_df.columns for col in feature_columns + target_columns):
            print("FATAL ERROR: Not all expected feature/target columns found in DataFrame after processing.")
            print("Features expected (start with): p0_, p1_, parton_")
            print("Targets expected:", target_columns)
            print("Columns found:", feature_target_df.columns.tolist())
            exit()

        print(
            f"\nUsing {len(feature_columns)} Feature Columns: {feature_columns[:5]}...{feature_columns[-5:]}")
        print(f"Using {len(target_columns)} Target Columns: {target_columns}")

        sample_size = 5000

        if sample_size < len(feature_target_df):
            print(f"\nSampling {sample_size} events for faster execution...")
            feature_target_df_sample = feature_target_df.sample(n=sample_size, random_state=42)
        else:
            print(f"\nUsing full dataset ({len(feature_target_df)} events).")
            feature_target_df_sample = feature_target_df

        # --- Prepare and Scale Data ---
        X_scaled, y_scaled, scaler_x, scaler_y = prepare_data_regression(
            feature_target_df_sample, feature_columns, target_columns
        )

        if X_scaled is not None and y_scaled is not None:
            # Split Data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled, test_size=0.25, random_state=42  # Use 25% test split
            )
            feature_dimension = X_train.shape[1]
            print(f"\nData split: {X_train.shape[0]} train, {X_test.shape[0]} test samples.")
            print(f"Parton feature dimension: {feature_dimension}")

            # --- Run Classical Models ---
            best_rf_model, y_pred_rf_tuned_scaled = main_regression_classical(
                X_train, X_test, y_train, y_test
            )

            # --- Run Quantum Model ---
            y_pred_quantum_scaled = None
            use_amplitude = False
            max_allowed_qubits = 12
            required_qubits = int(np.ceil(np.log2(feature_dimension))) if use_amplitude else feature_dimension

            if required_qubits <= max_allowed_qubits:
                y_pred_quantum_scaled, _, _ = main_regression_quantum_pennylane_fixed(
                    X_train, X_test, y_train, y_test,
                    feature_dimension=feature_dimension,
                    use_amplitude_embedding=use_amplitude
                )
            else:
                print(f"\n--- Skipping Quantum Kernel Regression ---")
                print(f"Required qubits ({required_qubits}) exceeds allowed limit ({max_allowed_qubits}).")

            print("\n--- Inverse Transforming Data for KL Evaluation ---")
            y_test_orig = scaler_y.inverse_transform(y_test)
            y_pred_rf_tuned_orig = scaler_y.inverse_transform(y_pred_rf_tuned_scaled)

            print("\nEvaluating Tuned RandomForest:")
            kl_scores_rf, final_kl_rf = evaluate_kl_divergence(y_test_orig, y_pred_rf_tuned_orig)

            if y_pred_quantum_scaled is not None:
                y_pred_quantum_orig = scaler_y.inverse_transform(y_pred_quantum_scaled)
                print("\nEvaluating Quantum Kernel SVR:")
                kl_scores_q, final_kl_q = evaluate_kl_divergence(y_test_orig, y_pred_quantum_orig)
            print("\n--- Generating Submission File (Example using Tuned RF) ---")
            test_data_fp = os.path.join(data_path, 'pp-z-to-jets-500K-54167.h5')
            submission_df = None
            if os.path.exists(test_data_fp):
                try:
                    with h5py.File(test_data_fp, 'r') as f_test:
                        print(f_test.keys())
                        if 'partons' in f_test:
                            partons_test_data = f_test['partons'][:]
                            print(f"Loaded test parton data. Shape: {partons_test_data.shape}")
                            num_test_events = partons_test_data.shape[0]
                            test_column_names = [f'parton_{i}_{comp}' for i in range(2) for comp in
                                                 ['px', 'py', 'pz', 'E', 'id', 'charge']]
                            partons_test_raw_df = pd.DataFrame(partons_test_data.reshape(num_test_events, -1),
                                                               columns=test_column_names)

                            test_features_df = extract_parton_features(partons_test_raw_df)
                            print(test_features_df.columns)
                            test_features_df = test_features_df.reindex(columns=feature_columns,
                                                                        fill_value=0)
                            print("print after reindex")
                            print(test_features_df.columns)
                            test_features_df['event_id'] = test_features_df.index
                            X_test_final_raw = test_features_df[feature_columns].values
                            if np.any(~np.isfinite(X_test_final_raw)):
                                print(
                                    "Warning: Non-finite values found in test feature data before scaling. Replacing with 0.")
                                X_test_final_raw = np.nan_to_num(X_test_final_raw, nan=0.0, posinf=0.0, neginf=0.0)
                            X_test_final_scaled = scaler_x.transform(
                                X_test_final_raw)

                            print(f"Predicting on {len(X_test_final_scaled)} test samples...")
                            y_pred_final_scaled = best_rf_model.predict(X_test_final_scaled)

                            y_pred_final_orig = scaler_y.inverse_transform(y_pred_final_scaled)

                            submission_df = pd.DataFrame()
                            submission_df['EventID'] = test_features_df['event_id']  # Assumes event_id was kept
                            submission_df['n_jets_pred'] = np.round(y_pred_final_orig[:, 0]).astype(int)
                            submission_df['n_jets_pred'] = submission_df['n_jets_pred'].clip(lower=0)
                            submission_df['leading_pt_pred'] = np.maximum(0.0, y_pred_final_orig[:, 1])
                            submission_df['subleading_pt_pred'] = np.maximum(0.0, y_pred_final_orig[:, 2])
                            submission_df.loc[submission_df['n_jets_pred'] < 2, 'subleading_pt_pred'] = 0.0

                            submission_filename = os.path.join(output_path, 'submission.csv')
                            submission_df.to_csv(submission_filename, index=False)
                            print(f"Submission file saved to: {submission_filename}")

                        else:
                            print(f"Error: 'partons' key not found in test file {test_data_fp}")
                except FileNotFoundError:
                    print(f"Error: Test data file not found at {test_data_fp}")
                except Exception as e:
                    print(f"Error during test data processing or prediction: {e}")
                    traceback.print_exc()
            else:
                print(f"Test data file not found at {test_data_fp}. Skipping submission file generation.")


        else:
            print("\nError preparing data for regression. Cannot proceed.")
    else:
        print("\nFailed to process training data. Cannot proceed.")

    main_end_time = time.time()
    print(f"\nTotal script execution time: {main_end_time - main_start_time:.2f} seconds.")

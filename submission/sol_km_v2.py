import os
import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import pennylane as qml
from tqdm import tqdm
import io


class LorentzVector:
    def __init__(self, px, py, pz, E):
        self.px = px
        self.py = py
        self.pz = pz
        self.E = E

    def pt(self):
        return np.sqrt(self.px ** 2 + self.py ** 2)

    def eta(self):
        p = np.sqrt(self.px ** 2 + self.py ** 2 + self.pz ** 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            return 0.5 * np.log((p + self.pz) / (p - self.pz))

    def phi(self):
        return np.arctan2(self.py, self.px)

    def inv_mass(self):
        return np.sqrt(self.E ** 2 - self.px ** 2 - self.py ** 2 - self.pz ** 2)


def compute_derived_quantities_partons(df):
    results = []
    for index, row in df.iterrows():
        parton_1 = LorentzVector(px=row['parton_0_px'], py=row['parton_0_py'], pz=row['parton_0_pz'],
                                 E=row['parton_0_E'])
        parton_2 = LorentzVector(px=row['parton_1_px'], py=row['parton_1_py'], pz=row['parton_1_pz'],
                                 E=row['parton_1_E'])
        pt_1 = parton_1.pt()
        pt_2 = parton_2.pt()
        eta_1 = parton_1.eta()
        eta_2 = parton_2.eta()
        phi_1 = parton_1.phi()
        phi_2 = parton_2.phi()

        parton_parton_delta_r = compute_delta_r(eta_1, phi_1, eta_2, phi_2)
        results.append({
            'event_id': index,
            'parton_0_pt': pt_1,
            'parton_1_pt': pt_2,
            'parton_0_eta': eta_1,
            'parton_1_eta': eta_2,
            'parton_0_phi': phi_1,
            'parton_1_phi': phi_2,
            'parton_parton_delta_r': parton_parton_delta_r
        })
    return pd.DataFrame(results)


def compute_derived_quantities_jets(df, num_max_jets):
    results = []
    for index, row in df.iterrows():
        jet_list = []
        for j in range(num_max_jets):
            if row[f'jet_{j}_px'] == 0 and row[f'jet_{j}_py'] == 0 and row[f'jet_{j}_pz'] == 0 and row[
                f'jet_{j}_E'] == 0:
                continue
            jet = LorentzVector(px=row[f'jet_{j}_px'], py=row[f'jet_{j}_py'], pz=row[f'jet_{j}_pz'],
                                E=row[f'jet_{j}_E'])
            jet_list.append({
                'pt': jet.pt(),
                'eta': jet.eta(),
                'phi': jet.phi(),
                'E': jet.E,
                'index': j
            })
        jet_list.sort(key=lambda x: x['pt'], reverse=True)
        leading_jet = jet_list[0] if len(jet_list) > 0 else {}
        subleading_jet = jet_list[1] if len(jet_list) > 1 else {}

        results.append({
            'event_id': index,
            'leading_jet_pt': leading_jet.get('pt', 0),
            'leading_jet_eta': leading_jet.get('eta', 0),
            'leading_jet_phi': leading_jet.get('phi', 0),
            'leading_jet_E': leading_jet.get('E', 0),
            'subleading_jet_pt': subleading_jet.get('pt', 0),
            'subleading_jet_eta': subleading_jet.get('eta', 0),
            'subleading_jet_phi': subleading_jet.get('phi', 0),
            'subleading_jet_E': subleading_jet.get('E', 0)
        })
    return pd.DataFrame(results)


def compute_delta_r(eta1, phi1, eta2, phi2):
    delta_eta = eta1 - eta2
    delta_phi = phi1 - phi2
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    delta_r = np.sqrt(delta_eta ** 2 + delta_phi ** 2)
    return delta_r


def compute_delta_r_vectorized(eta1, phi1, eta2, phi2):
    delta_eta = eta1[:, np.newaxis] - eta2
    delta_phi = phi1[:, np.newaxis] - phi2
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(delta_eta ** 2 + delta_phi ** 2)


def process_hep_data(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Processing {file_path}")
            data_file_name = file_path.split('/')[-1]
            name, ext = os.path.splitext(data_file_name)
            data_file = name.split('-')[-1]
            print(f'Keys {f.keys()}')

            partons_df = None
            jets_df = None

            if 'partons' in f:
                print(f"Processing partons {data_file}")
                partons_data = f['partons'][:]
                column_names = [f'parton_{i}_{component}' for i in range(partons_data.shape[1]) for component in
                                ['px', 'py', 'pz', 'E', 'id', 'charge']]
                partons_df = pd.DataFrame(partons_data.reshape(partons_data.shape[0], -1), columns=column_names)
                partons_derived_df = compute_derived_quantities_partons(partons_df)
                partons_df['event_id'] = np.arange(partons_data.shape[0])
                partons_df = pd.merge(partons_df, partons_derived_df, on='event_id')
                print("Partons DataFrame with Derived Quantities: processed")

            if 'jets' in f:
                print(f"Processing jets {data_file}")
                jets_data = f['jets'][:]
                num_max_jets = jets_data.shape[1]
                jet_column_names = [f'jet_{j}_{component}' for j in range(num_max_jets) for component in
                                    ['px', 'py', 'pz', 'E']]
                jets_df = pd.DataFrame(jets_data.reshape(jets_data.shape[0], -1), columns=jet_column_names)
                jets_derived_df = compute_derived_quantities_jets(jets_df, num_max_jets)
                jets_df['event_id'] = np.arange(jets_df.shape[0])
                jets_df = pd.merge(jets_df, jets_derived_df, on='event_id')
                print("Jets DataFrame with Leading/Subleading Jet Info: processed")

            if partons_df is not None and jets_df is not None:
                matched_df = match_jets_to_partons(jets_df, partons_df)
                return matched_df
            else:
                return None

    except h5py.FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None


def match_jets_to_partons(jets_df, partons_df, delta_r_threshold=0.3, batch_size=1000):
    print("Starting jet-parton matching for regression...")
    print(f"Initial number of events in jets_df: {len(jets_df)}")
    print(f"Initial number of events in partons_df: {len(partons_df)}")
    matches = []

    jet_eta = jets_df['leading_jet_eta'].values
    jet_phi = jets_df['leading_jet_phi'].values
    jet_pt = jets_df['leading_jet_pt'].values
    jet_event_ids = jets_df['event_id'].values

    parton_0_eta = partons_df['parton_0_eta'].values
    parton_0_phi = partons_df['parton_0_phi'].values
    parton_0_pt = partons_df['parton_0_pt'].values
    parton_0_event_ids = partons_df['event_id'].values

    parton_1_eta = partons_df['parton_1_eta'].values
    parton_1_phi = partons_df['parton_1_phi'].values
    parton_1_pt = partons_df['parton_1_pt'].values
    parton_1_event_ids = partons_df['event_id'].values

    for start_idx in tqdm(range(0, len(jets_df), batch_size), desc="Processing batches for matching"):
        end_idx = min(start_idx + batch_size, len(jets_df))

        batch_jet_eta = jet_eta[start_idx:end_idx]
        batch_jet_phi = jet_phi[start_idx:end_idx]
        batch_jet_pt = jet_pt[start_idx:end_idx]
        batch_jet_event_ids = jet_event_ids[start_idx:end_idx]

        batch_jet_parton0_delta_r = compute_delta_r_vectorized(
            batch_jet_eta, batch_jet_phi,
            parton_0_eta, parton_0_phi
        )
        batch_jet_parton1_delta_r = compute_delta_r_vectorized(
            batch_jet_eta, batch_jet_phi,
            parton_1_eta, parton_1_phi
        )

        for i in range(len(batch_jet_eta)):
            best_match_dr = float('inf')
            matched_parton = -1

            min_index_0 = np.argmin(batch_jet_parton0_delta_r[i])
            if batch_jet_parton0_delta_r[i, min_index_0] < delta_r_threshold:
                if batch_jet_parton0_delta_r[i, min_index_0] < best_match_dr:
                    best_match_dr = batch_jet_parton0_delta_r[i, min_index_0]
                    matched_parton = 0

            min_index_1 = np.argmin(batch_jet_parton1_delta_r[i])
            if batch_jet_parton1_delta_r[i, min_index_1] < delta_r_threshold:
                if batch_jet_parton1_delta_r[i, min_index_1] < best_match_dr:
                    best_match_dr = batch_jet_parton1_delta_r[i, min_index_1]
                    matched_parton = 1

            if matched_parton != -1:
                if matched_parton == 0:
                    matches.append({
                        'event_id': batch_jet_event_ids[i],
                        'parton_0_pt': parton_0_pt[min_index_0],
                        'parton_0_eta': parton_0_eta[min_index_0],
                        'parton_0_phi': parton_0_phi[min_index_0],
                        'leading_jet_pt': batch_jet_pt[i],
                        'leading_jet_eta': batch_jet_eta[i],
                        'leading_jet_phi': batch_jet_phi[i],
                        'leading_jet_E':
                            jets_df.loc[jets_df['event_id'] == batch_jet_event_ids[i]]['leading_jet_E'].values[0],
                        'subleading_jet_pt':
                            jets_df.loc[jets_df['event_id'] == batch_jet_event_ids[i]]['subleading_jet_pt'].values[0],
                        'subleading_jet_eta':
                            jets_df.loc[jets_df['event_id'] == batch_jet_event_ids[i]]['subleading_jet_eta'].values[0],
                        'subleading_jet_phi':
                            jets_df.loc[jets_df['event_id'] == batch_jet_event_ids[i]]['subleading_jet_phi'].values[0],
                        'subleading_jet_E':
                            jets_df.loc[jets_df['event_id'] == batch_jet_event_ids[i]]['subleading_jet_E'].values[0],
                        'delta_r_leading': best_match_dr,
                        'matched_parton': matched_parton
                    })
                elif matched_parton == 1:
                    matches.append({
                        'event_id': batch_jet_event_ids[i],
                        'parton_0_pt': parton_0_pt[np.argmin(batch_jet_parton0_delta_r[i])],
                        'parton_0_eta': parton_0_eta[np.argmin(batch_jet_parton0_delta_r[i])],
                        'parton_0_phi': parton_0_phi[np.argmin(batch_jet_parton0_delta_r[i])],
                        'parton_1_pt': parton_1_pt[min_index_1],
                        'parton_1_eta': parton_1_eta[min_index_1],
                        'parton_1_phi': parton_1_phi[min_index_1],
                        'leading_jet_pt': batch_jet_pt[i],
                        'leading_jet_eta': batch_jet_eta[i],
                        'leading_jet_phi': batch_jet_phi[i],
                        'leading_jet_E':
                            jets_df.loc[jets_df['event_id'] == batch_jet_event_ids[i]]['leading_jet_E'].values[0],
                        'subleading_jet_pt':
                            jets_df.loc[jets_df['event_id'] == batch_jet_event_ids[i]]['subleading_jet_pt'].values[0],
                        'subleading_jet_eta':
                            jets_df.loc[jets_df['event_id'] == batch_jet_event_ids[i]]['subleading_jet_eta'].values[0],
                        'subleading_jet_phi':
                            jets_df.loc[jets_df['event_id'] == batch_jet_event_ids[i]]['subleading_jet_phi'].values[0],
                        'subleading_jet_E':
                            jets_df.loc[jets_df['event_id'] == batch_jet_event_ids[i]]['subleading_jet_E'].values[0],
                        'delta_r_leading': best_match_dr,
                        'matched_parton': matched_parton
                    })

    matched_df = pd.DataFrame(matches)
    print(f"Number of matched events: {len(matched_df)}")
    return matched_df


def prepare_data_regression(matched_df, feature_columns, target_columns):
    if matched_df is None:
        return None, None, None

    X = matched_df[feature_columns].values
    y = matched_df[target_columns].values
    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    return X_scaled, y_scaled, scaler_y


import pennylane as qml


from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor

def main_regression_quantum_pennylane_fixed(X_train, X_test, y_train, y_test, feature_dimension, use_amplitude_embedding=False):
    """
    Runs Quantum Kernel Regression using PennyLane, with options for feature map and hyperparameter tuning.
    """
    n_qubits = feature_dimension
    dev = qml.device("default.qubit", wires=n_qubits, shots=None)

    def angle_embedding(features):
        for i in range(n_qubits):
            qml.RY(np.pi * features[i], wires=i)
            qml.RZ(np.pi * features[i], wires=i)

    def amplitude_embedding(features):

        norm = np.linalg.norm(features)
        if norm == 0:
            features_normalized = np.zeros_like(features)
        else:
            features_normalized = features / norm
        qml.AmplitudeEmbedding(features_normalized, wires=range(n_qubits), pad_with=0, normalize=True)

    @qml.qnode(dev)
    def quantum_feature_map(features):
        if use_amplitude_embedding:
            amplitude_embedding(features)
        else:
            angle_embedding(features)
        return qml.state()

    def compute_quantum_kernel_pennylane(X):
        n_samples = len(X)
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in tqdm(range(n_samples), desc="Computing Quantum Kernel (Train)"):
            state_i = quantum_feature_map(X[i])
            for j in range(i, n_samples):
                state_j = quantum_feature_map(X[j])
                overlap = np.abs(np.vdot(np.conjugate(state_i), state_j)) ** 2
                kernel_matrix[i, j] = overlap
                kernel_matrix[j, i] = overlap
        return kernel_matrix

    def compute_test_quantum_kernel_pennylane(X_test, X_train):
        n_test = len(X_test)
        n_train = len(X_train)
        kernel_matrix_test = np.zeros((n_test, n_train))
        for i in tqdm(range(n_test), desc="Computing Quantum Kernel (Test)"):
            state_test = quantum_feature_map(X_test[i])
            for j in range(n_train):
                state_train = quantum_feature_map(X_train[j])
                overlap = np.abs(np.vdot(np.conjugate(state_test), state_train)) ** 2
                kernel_matrix_test[i, j] = overlap
        return kernel_matrix_test

    print("Computing Quantum Kernel for training data using PennyLane...")
    quantum_kernel_train = compute_quantum_kernel_pennylane(X_train)

    print("Training Quantum Kernel SVR with Hyperparameter Tuning...")
    param_grid = {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1, 0.2]}
    qsvr = SVR(kernel='precomputed')
    multi_output_qsvr = MultiOutputRegressor(GridSearchCV(qsvr, param_grid, cv=2, verbose=1, n_jobs=-1))
    multi_output_qsvr.fit(quantum_kernel_train, y_train)

    print("Best hyperparameters found:", multi_output_qsvr.estimators_[0].best_params_)

    print("Computing Quantum Kernel for test data using PennyLane...")
    quantum_kernel_test = compute_test_quantum_kernel_pennylane(X_test, X_train)

    print("Evaluating Quantum Kernel SVR...")
    y_pred_quantum = multi_output_qsvr.predict(quantum_kernel_test)

    mse_quantum = mean_squared_error(y_test, y_pred_quantum)
    r2_quantum = r2_score(y_test, y_pred_quantum, multioutput='uniform_average')

    print(f'Quantum Kernel SVR (PennyLane) - Mean Squared Error: {mse_quantum:.4f}')
    print(f'Quantum Kernel SVR (PennyLane) - R-squared: {r2_quantum:.4f}')

def main_regression_classical(X_train, X_test, y_train, y_test):


    print("Training Classical RandomForestRegressor...")

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    y_pred_classical = model.predict(X_test)

    mse_classical = mean_squared_error(y_test, y_pred_classical)

    r2_classical = r2_score(y_test, y_pred_classical)

    print(f'Classical RandomForestRegressor - Mean Squared Error: {mse_classical:.4f}')

    print(f'Classical RandomForestRegressor - R-squared: {r2_classical:.4f}')

if __name__ == "__main__":
    base_path = os.getcwd()
    data_path = base_path.replace('submission', 'data')
    output_path = base_path.replace('submission', 'datasets')
    jets_partons_fp = f'{data_path}/pp-z-to-jets-500K-57246.h5'

    matched_df = process_hep_data(jets_partons_fp)

    print(f" columns in matched_df: {matched_df.columns.tolist() if matched_df is not None else None}")

    if matched_df is not None:
        sample_size = 1000
        matched_df = matched_df.sample(n=sample_size, random_state=42).reset_index()
        feature_columns_regression = [
            'parton_0_pt',
            'parton_0_eta',
            'parton_0_phi',
            'leading_jet_pt',
            'leading_jet_eta',
            'leading_jet_phi',
            'delta_r_leading',
            'matched_parton'
        ]
        target_columns_regression = [
            'leading_jet_pt', 'subleading_jet_pt'
        ]

        print("\nPreparing data for regression...")
        X, y, _ = prepare_data_regression(matched_df, feature_columns_regression, target_columns_regression)
        if X is not None and y is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            feature_dimension = X_train.shape[1]

            print("\n--- Running Classical Regression on Matched Data ---")
            main_regression_classical(X_train, X_test, y_train, y_test)

            print("\n--- Running Hyperparameter Tuning for Classical Regression ---")
            param_grid = {
                'n_estimators': [50, 100, 150, 200, 250, 300],
                'max_depth': [8, 10, 12, 14, None],
                'min_samples_split': [2, 3, 4, 5],
                'min_samples_leaf': [1, 2, 3, 4],
                'bootstrap': [True, False]
            }
            rf = RandomForestRegressor(random_state=42)
            random_search = RandomizedSearchCV(estimator=rf,
                                               param_distributions=param_grid,
                                               n_iter=5,
                                               cv=2,
                                               verbose=1,
                                               n_jobs=-1,
                                               random_state=42)
            random_search.fit(X_train, y_train)
            best_params = random_search.best_params_
            best_score = random_search.best_score_
            print(f"Best parameters found: {best_params}")
            print(f"Best R-squared score on training data: {best_score:.4f}")

            best_rf_model = RandomForestRegressor(**best_params, random_state=42)
            best_rf_model.fit(X_train, y_train)
            y_pred_tuned = best_rf_model.predict(X_test)
            mse_tuned = mean_squared_error(y_test, y_pred_tuned)
            r2_tuned = r2_score(y_test, y_pred_tuned)

            print(f'Tuned Classical RandomForestRegressor - Mean Squared Error: {mse_tuned:.4f}')
            print(f'Tuned Classical RandomForestRegressor - R-squared: {r2_tuned:.4f}')

            print("\n--- Running Quantum Kernel Regression on Matched Data (PennyLane) ---")
            main_regression_quantum_pennylane_fixed(X_train, X_test, y_train, y_test, feature_dimension)

        else:
            print("Error preparing data for regression.")
    else:
        print("Some required columns are missing in the matched DataFrame for the regression task.")

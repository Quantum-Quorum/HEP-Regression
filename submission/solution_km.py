import os
import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from tqdm import tqdm
import io

class LorentzVector:
    def __init__(self, px, py, pz, E):
        self.px = px
        self.py = py
        self.pz = pz
        self.E = E
    
    def pt(self):
        return np.sqrt(self.px**2 + self.py**2)
    
    def eta(self):
        p = np.sqrt(self.px**2 + self.py**2 + self.pz**2)
        return 0.5 * np.log((p + self.pz) / (p - self.pz))
    
    def phi(self):
        return np.arctan2(self.py, self.px)
    
    def inv_mass(self):
        return np.sqrt(self.E - self.px**2 + self.py**2 + self.pz**2)

def compute_derived_quantities_partons(df):
    """
    Compute derived quantities for partons including transverse momentum, pseudorapidity,
    azimuthal angle, and the angular separation between the two partons.
    """
    results = []
    for index, row in df.iterrows():
        parton_1 = LorentzVector(px=row['parton_0_px'], py=row['parton_0_py'], pz=row['parton_0_pz'], E=row['parton_0_E'])
        parton_2 = LorentzVector(px=row['parton_1_px'], py=row['parton_1_py'], pz=row['parton_1_pz'], E=row['parton_1_E'])
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
        jet_data = []
        for j in range(num_max_jets):
            if row[f'jet_{j}_px'] == 0 and row[f'jet_{j}_py'] == 0 and row[f'jet_{j}_pz'] == 0 and row[f'jet_{j}_E'] == 0:
                continue 
            jet = LorentzVector(px=row[f'jet_{j}_px'], py=row[f'jet_{j}_py'], pz=row[f'jet_{j}_pz'], E=row[f'jet_{j}_E'])
            jet_data.append({
                f'jet_{j}_pt': jet.pt(),
                f'jet_{j}_eta': jet.eta(),
                f'jet_{j}_phi': jet.phi()
            })
        results.append({'event_id': index, **{k: v for d in jet_data for k, v in d.items()}})
    return pd.DataFrame(results)

def compute_delta_r(eta1, phi1, eta2, phi2):
    """
    Compute the angular distance between two particles in eta-phi space.
    
    Args:
        eta1 (float): Pseudorapidity of first particle
        phi1 (float): Azimuthal angle of first particle
        eta2 (float): Pseudorapidity of second particle
        phi2 (float): Azimuthal angle of second particle
        
    Returns:
        float: Delta R between the particles
    """
    delta_eta = eta1 - eta2
    delta_phi = phi1 - phi2
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    delta_r = np.sqrt(delta_eta**2 + delta_phi**2)
    return delta_r

def compute_delta_r_vectorized(eta1, phi1, eta2, phi2):
    delta_eta = eta1[:, np.newaxis] - eta2
    delta_phi = phi1[:, np.newaxis] - phi2
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(delta_eta**2 + delta_phi**2)

def process_hep_data(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Processing {file_path}")
            data_file_name = file_path.split('/')[-1]
            data_file = data_file_name.split('-')[-1]
            print(f'Keys {f.keys()}')
            
            if 'partons' in f and 'jets' in f:
                print(f"Processing partons and jets {data_file}")
                partons_data = f['partons'][:]
                column_names = [f'parton_{i}_{component}' for i in range(partons_data.shape[1]) for component in ['px', 'py', 'pz', 'E', 'id', 'charge']]
                partons_df = pd.DataFrame(partons_data.reshape(partons_data.shape[0], -1), columns=column_names)
                partons_derived_df = compute_derived_quantities_partons(partons_df)
                partons_df['event_id'] = np.arange(partons_data.shape[0])
                partons_df = pd.merge(partons_df, partons_derived_df, on='event_id')
                partons_df.to_csv(f"{output_path}/{data_file}_partons_df.csv")
                print("Partons DataFrame with Derived Quantities: exported")
                
                jets_data = f['jets'][:]
                num_max_jets = jets_data.shape[1]
                jet_column_names = [f'jet_{j}_{component}' for j in range(num_max_jets) for component in ['px', 'py', 'pz', 'E']]
                jets_df = pd.DataFrame(jets_data.reshape(jets_data.shape[0], -1), columns=jet_column_names)
                jets_derived_df = compute_derived_quantities_jets(jets_df, num_max_jets)
                jets_df['event_id'] = np.arange(jets_df.shape[0])
                jets_df = pd.merge(jets_df, jets_derived_df, on='event_id')
                jets_df.to_csv(f"{output_path}/{data_file}_jets_df.csv")
                print("Jets DataFrame with Derived Quantities: exported")
                
                merged_df = pd.merge(partons_df, jets_df, on='event_id')
                return partons_df, jets_df
                
            elif 'partons' in f:
                print(f"Processing partons {data_file}")
                partons_data = f['partons'][:]
                column_names = [f'parton_{i}_{component}' for i in range(partons_data.shape[1]) for component in ['px', 'py', 'pz', 'E', 'id', 'charge']]
                partons_df = pd.DataFrame(partons_data.reshape(partons_data.shape[0], -1), columns=column_names)
                partons_derived_df = compute_derived_quantities_partons(partons_df)
                partons_df['event_id'] = np.arange(partons_data.shape[0])
                partons_df = pd.merge(partons_df, partons_derived_df, on='event_id')
                partons_df.to_csv(f"{output_path}/{data_file}_partons_df.csv")
                print("Partons DataFrame with Derived Quantities:")
                
            elif 'jets' in f:
                print(f"Processing jets {data_file}")
                jets_data = f['jets'][:]
                num_max_jets = jets_data.shape[1]
                jet_column_names = [f'jet_{j}_{component}' for j in range(num_max_jets) for component in ['px', 'py', 'pz', 'E']]
                jets_df = pd.DataFrame(jets_data.reshape(jets_data.shape[0], -1), columns=jet_column_names)
                jets_derived_df = compute_derived_quantities_jets(jets_df, num_max_jets)
                jets_df['event_id'] = np.arange(jets_df.shape[0])
                jets_df = pd.merge(jets_df, jets_derived_df, on='event_id')
                jets_df.to_csv(f"{output_path}/{data_file}_jets_df.csv")
                print("Jets DataFrame with Derived Quantities:")
                
            else:
                print("Neither 'partons' nor 'jets' dataset found in the file.")
    except h5py.FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None, None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None, None

def match_jets_to_partons(jets_df, partons_df, delta_r_threshold=0.3, batch_size=1000):
    """
    Match jets to their corresponding partons using angular separation (delta R).
    This function calculates the angular separation between jets and partons to identify
    which jet corresponds to which parton.
    
    Args:
        jets_df: DataFrame containing jet information
        partons_df: DataFrame containing parton information
        delta_r_threshold: Maximum allowed delta R for matching
        batch_size: Number of jets to process at once to avoid memory issues
    """
    print("Starting jet-parton matching...")
    print(f"Initial number of events: {len(jets_df)}")
    matches = []
    
    
    jet_eta = jets_df['jet_0_eta'].values
    jet_phi = jets_df['jet_0_phi'].values
    jet_pt = jets_df['jet_0_pt'].values
    jet_event_ids = jets_df['event_id'].values
    
    parton_0_eta = partons_df['parton_0_eta'].values
    parton_0_phi = partons_df['parton_0_phi'].values
    parton_0_pt = partons_df['parton_0_pt'].values
    parton_0_event_ids = partons_df['event_id'].values
    
    parton_1_eta = partons_df['parton_1_eta'].values
    parton_1_phi = partons_df['parton_1_phi'].values
    parton_1_pt = partons_df['parton_1_pt'].values
    parton_1_event_ids = partons_df['event_id'].values
    
   
    total_matches = 0
    matches_to_parton0 = 0
    matches_to_parton1 = 0
    unmatched_jets = 0
    
  
    delta_r_values = []
    pt_ratios = []
    
   
    for start_idx in tqdm(range(0, len(jets_df), batch_size), desc="Processing batches"):
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
            matched = False
            
            
            min_index_0 = np.argmin(batch_jet_parton0_delta_r[i])
            if batch_jet_parton0_delta_r[i, min_index_0] < delta_r_threshold:
                pt_ratio = min(batch_jet_pt[i], parton_0_pt[min_index_0]) / max(batch_jet_pt[i], parton_0_pt[min_index_0])
                matches.append({
                    'jet_id': batch_jet_event_ids[i],
                    'parton_id': parton_0_event_ids[min_index_0],
                    'delta_r': batch_jet_parton0_delta_r[i, min_index_0],
                    'jet_pt': batch_jet_pt[i],
                    'jet_eta': batch_jet_eta[i],
                    'jet_phi': batch_jet_phi[i],
                    'parton_pt': parton_0_pt[min_index_0],
                    'parton_eta': parton_0_eta[min_index_0],
                    'parton_phi': parton_0_phi[min_index_0],
                    'parton_number': 0,
                    'pt_ratio': pt_ratio
                })
                matches_to_parton0 += 1
                matched = True
                delta_r_values.append(batch_jet_parton0_delta_r[i, min_index_0])
                pt_ratios.append(pt_ratio)
            
            min_index_1 = np.argmin(batch_jet_parton1_delta_r[i])
            if batch_jet_parton1_delta_r[i, min_index_1] < delta_r_threshold:
                pt_ratio = min(batch_jet_pt[i], parton_1_pt[min_index_1]) / max(batch_jet_pt[i], parton_1_pt[min_index_1])
                matches.append({
                    'jet_id': batch_jet_event_ids[i],
                    'parton_id': parton_1_event_ids[min_index_1],
                    'delta_r': batch_jet_parton1_delta_r[i, min_index_1],
                    'jet_pt': batch_jet_pt[i],
                    'jet_eta': batch_jet_eta[i],
                    'jet_phi': batch_jet_phi[i],
                    'parton_pt': parton_1_pt[min_index_1],
                    'parton_eta': parton_1_eta[min_index_1],
                    'parton_phi': parton_1_phi[min_index_1],
                    'parton_number': 1,
                    'pt_ratio': pt_ratio
                })
                matches_to_parton1 += 1
                matched = True
                delta_r_values.append(batch_jet_parton1_delta_r[i, min_index_1])
                pt_ratios.append(pt_ratio)
            
            if not matched:
                unmatched_jets += 1
    
    matches_df = pd.DataFrame(matches)
    total_matches = len(matches_df)
    
    delta_r_values = np.array(delta_r_values)
    pt_ratios = np.array(pt_ratios)


    # Ai assited to write below code
    print("Matched events Statistics:")
    print(f"Total number of events: {len(jets_df)}")
    print(f"Number of matched events: {total_matches}")
    print(f"Number of unmatched events: {unmatched_jets}")
    print(f"Matches to parton 0: {matches_to_parton0}")
    print(f"Matches to parton 1: {matches_to_parton1}")
    print(f"Reduction factor: {total_matches/len(jets_df):.2%}")
    print("\nDelta R Statistics:")
    print(f"Mean Delta R: {np.mean(delta_r_values):.3f}")
    print(f"Median Delta R: {np.median(delta_r_values):.3f}")
    print(f"Std Delta R: {np.std(delta_r_values):.3f}")
    print("\nPT Ratio Statistics:")
    print(f"Mean PT Ratio: {np.mean(pt_ratios):.3f}")
    print(f"Median PT Ratio: {np.median(pt_ratios):.3f}")
    print(f"Std PT Ratio: {np.std(pt_ratios):.3f}")
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = output_dir.replace('submission', 'datasets')
    os.makedirs(output_dir, exist_ok=True)
    
    matches_df.to_csv(f"{output_dir}/matched_events.csv", index=False)
    
    stats = {
        'total_events': len(jets_df),
        'matched_events': total_matches,
        'unmatched_events': unmatched_jets,
        'matches_to_parton0': matches_to_parton0,
        'matches_to_parton1': matches_to_parton1,
        'reduction_factor': total_matches/len(jets_df),
        'mean_delta_r': np.mean(delta_r_values),
        'median_delta_r': np.median(delta_r_values),
        'std_delta_r': np.std(delta_r_values),
        'mean_pt_ratio': np.mean(pt_ratios),
        'median_pt_ratio': np.median(pt_ratios),
        'std_pt_ratio': np.std(pt_ratios)
    }
    pd.DataFrame([stats]).to_csv(f"{output_dir}/matching_statistics.csv", index=False)
    
    return matches_df

def prepare_data(matched_df, feature_columns, label_column):
    X = matched_df[feature_columns].values
    y = matched_df[label_column].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def create_quantum_instance():
    return LocalSimulator()

def build_feature_map(feature_dimension):
    circuit = Circuit()
    for i in range(feature_dimension):
        circuit.ry(i, np.pi * feature_values[i])  
        circuit.rz(i, np.pi * feature_values[i])  
    return circuit

def compute_quantum_kernel(X_train, feature_map, quantum_instance):
    kernel_matrix = np.zeros((X_train.shape[0], X_train.shape[0]))
    for i in tqdm(range(X_train.shape[0]), desc="Computing Quantum Kernel"):
        for j in range(X_train.shape[0]):
            circuit = feature_map()
            for k in range(X_train.shape[1]):
                if X_train[i, k] == 1:
                    circuit.x(k)
                if X_train[j, k] == 1:
                    circuit.x(k).adjoint()
            result = quantum_instance.run(circuit, shots=1000)
            kernel_matrix[i, j] = result.measurement_probabilities.get("0" * circuit.qubit_count, 0)
    return kernel_matrix

def train_qsvm(X_train, y_train, kernel_matrix):
    qsvm = SVC(kernel='precomputed')
    qsvm.fit(kernel_matrix, y_train)
    return qsvm

def evaluate_model(qsvm, X_test, y_test, feature_map, quantum_instance):
    kernel_matrix_test = compute_quantum_kernel(X_test, feature_map, quantum_instance)
    y_pred = qsvm.predict(kernel_matrix_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def main(matched_df, feature_columns, label_column):
    X, y = prepare_data(matched_df, feature_columns, label_column)
    X_train, X_test, y_train, y_test = split_data(X, y)
    quantum_instance = create_quantum_instance()
    feature_map = build_feature_map(feature_dimension=X_train.shape[1])
    kernel_matrix = compute_quantum_kernel(X_train, feature_map, quantum_instance)
    qsvm = train_qsvm(X_train, y_train, kernel_matrix)
    accuracy = evaluate_model(qsvm, X_test, y_test, feature_map, quantum_instance)
    print(f'Accuracy: {accuracy:.4f}')

def validate_kinematic_variables(df):
    if 'pt' in df.columns:
        if (df['pt'] < 0).any():
            raise ValueError("Negative pT values found")
    if 'eta' in df.columns:
        if (abs(df['eta']) > 10).any():
            raise ValueError("Unphysical eta values found")

if __name__ == "__main__":
    base_path = os.getcwd()
    data_path = base_path.replace('submission', 'data')
    output_path = base_path.replace('submission', 'datasets')
    jets_partons_fp = f'{data_path}/pp-z-to-jets-500K-57246.h5'
    
    partons_df, jets_df = process_hep_data(jets_partons_fp)
    print(f" columns for partons, {partons_df.columns.tolist()}")
    print(f" columns for jets, {jets_df.columns.tolist()}")
    
    if partons_df is not None and jets_df is not None:
        matched_df = match_jets_to_partons(jets_df, partons_df)
        feature_columns = [
            'jet_0_pt', 'jet_0_eta', 'jet_0_phi',
            'parton_0_pt', 'parton_0_eta', 'parton_0_phi',
            'parton_1_pt', 'parton_1_eta', 'parton_1_phi'
        ]
        label_column = 'event_id'
        
        if all(column in matched_df.columns for column in feature_columns + [label_column]):
            print("columns matched ")
            main(matched_df, feature_columns, label_column)
        else:
            print("Some feature or missing in the df.") 
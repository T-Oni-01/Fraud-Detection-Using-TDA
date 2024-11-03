import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy import stats
import kmapper as km
import matplotlib.pyplot as plt
import gudhi
from scipy.spatial.distance import pdist, squareform
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import webbrowser
from sklearn.cluster import DBSCAN


class AnomalyDetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Anomaly Detection Application")
        self.geometry("800x600")

        # Initialize file name variable and anomaly data
        self.file_name = None
        self.anomalies_df = None

        self.label = tk.Label(self, text="Enter CSV file name:")
        self.label.pack()

        self.file_entry = tk.Entry(self, width=50)
        self.file_entry.pack()

        self.console_output = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=80, height=20)
        self.console_output.pack()

        self.run_button = tk.Button(self, text="Run Analysis", command=self.run_analysis)
        self.run_button.pack()

        self.show_mapper_button = tk.Button(self, text="Show Mapper Output", command=self.show_mapper_output,
                                            state=tk.DISABLED)
        self.show_mapper_button.pack()

        self.show_anomalies_button = tk.Button(self, text="Show Anomalies", command=self.show_anomalies,
                                               state=tk.DISABLED)
        self.show_anomalies_button.pack()

    def run_analysis(self):
        csv_file = self.file_entry.get()

        if not csv_file:
            messagebox.showerror("Error", "Please enter a CSV file name.")
            return

        self.console_output.delete(1.0, tk.END)  # Clear previous output

        self.file_name = csv_file

        try:
            df_cleaned = self.load_and_clean_data(csv_file)
            if df_cleaned is None or df_cleaned.empty:
                raise ValueError("Data could not be cleaned or processed. Please check the file content and format.")

            self.console_output.insert(tk.END, "Data loaded and cleaned successfully.\n")

            data = df_cleaned[features].dropna().values

            scaler = StandardScaler()
            data = scaler.fit_transform(data)

            pca = PCA(n_components=3)
            data_reduced = pca.fit_transform(data)
            self.console_output.insert(tk.END, f"Shape of data after PCA: {data_reduced.shape}\n")

            # Z-score anomaly detection
            z_scores = stats.zscore(data_reduced)
            threshold = 2.2
            abs_z_scores = np.abs(z_scores)
            anomalies_mask_zscore = (abs_z_scores >= threshold).any(axis=1)

            df_cleaned['ZScore_Anomaly'] = anomalies_mask_zscore.astype(int)
            anomalies_df_zscore = df_cleaned[df_cleaned['ZScore_Anomaly'] == 1]
            self.console_output.insert(tk.END, f"Number of anomalies detected by Z-score: {len(anomalies_df_zscore)}\n")

            # Isolation Forest anomaly detection
            iso_forest = IsolationForest(contamination=0.01, random_state=42)
            iso_forest.fit(data_reduced)
            anomalies_mask_iso = iso_forest.predict(data_reduced) == -1
            df_cleaned['IsolationForest_Anomaly'] = anomalies_mask_iso.astype(int)
            anomalies_df_iso = df_cleaned[df_cleaned['IsolationForest_Anomaly'] == 1]
            self.console_output.insert(tk.END,
                                       f"Number of anomalies detected by Isolation Forest: {len(anomalies_df_iso)}\n")

            # Logistic Regression anomaly detection
            df_cleaned['Anomaly'] = 0
            df_cleaned.loc[anomalies_mask_zscore, 'Anomaly'] = 1
            df_cleaned.loc[anomalies_mask_iso, 'Anomaly'] = 1

            X = data_reduced
            y = df_cleaned['Anomaly'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            log_reg = LogisticRegression(max_iter=1000)
            log_reg.fit(X_train, y_train)

            y_prob_all = log_reg.predict_proba(X)[:, 1]
            custom_threshold = 0.1
            df_cleaned['Logistic_Anomaly'] = (y_prob_all >= custom_threshold).astype(int)

            self.console_output.insert(tk.END,
                                       f"Number of anomalies detected by Logistic Regression with threshold {custom_threshold}: {sum(df_cleaned['Logistic_Anomaly'])}\n")

            # Combine anomaly detections
            df_cleaned['Combined_Anomaly'] = df_cleaned[
                ['ZScore_Anomaly', 'IsolationForest_Anomaly', 'Logistic_Anomaly']].max(axis=1)
            num_combined_anomalies = df_cleaned['Combined_Anomaly'].sum()
            self.console_output.insert(tk.END,
                                       f"Number of anomalies detected by combined approach: {num_combined_anomalies}\n")

            # Save anomalies dataframe for later use
            self.anomalies_df = df_cleaned[df_cleaned['Combined_Anomaly'] == 1].copy()


            # Compute Persistent Homology
            distance_matrix = squareform(pdist(data_reduced))
            rips_complex = gudhi.RipsComplex(distance_matrix=distance_matrix)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
            pers_diag = simplex_tree.persistence()

            # Plot persistence diagram
            gudhi.plot_persistence_diagram(pers_diag)
            plt.title("Persistence Diagram")
            plt.show()

            # Generate Mapper Output
            mapper = km.KeplerMapper()
            lens = mapper.fit_transform(data_reduced, projection=PCA(n_components=2))
            graph = mapper.map(lens, data_reduced)

            # Save mapper output
            mapper.visualize(graph, path_html='mapper_output.html')
            self.console_output.insert(tk.END, "KeplerMapper output saved to 'mapper_output.html'.\n")

            # Integrate TDA results for Anomaly Detection
            tda_anomalies = self.detect_anomalies_with_tda(data_reduced, graph)
            self.console_output.insert(tk.END, f"Number of anomalies detected by TDA: {len(tda_anomalies)}\n")

            # Enable the new buttons
            self.show_mapper_button.config(state=tk.NORMAL)
            self.show_anomalies_button.config(state=tk.NORMAL)

            # Save anomalies dataframe for later use
            self.anomalies_df = df_cleaned[df_cleaned['Combined_Anomaly'] == 1].copy()

            # Plot anomalies
            self.plot_graphs(data_reduced, abs_z_scores, anomalies_mask_zscore, anomalies_mask_iso, df_cleaned)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_and_clean_data(self, file_path):
        global features
        try:
            df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
            self.console_output.insert(tk.END, "CSV data loaded successfully.\n")
        except UnicodeDecodeError as e:
            self.console_output.insert(tk.END, f"Error reading the CSV file: {e}\n")
            return None

        df.columns = df.columns.str.strip()

        column_mapping = {
            'Account No': 'Account No.',
            'DATE': 'Date',
            'TRANSACTION DETAILS': 'Transaction Details',
            'CHQ.NO.': 'Cheque No.',
            'VALUE DATE': 'Value Date',
            'WITHDRAWAL AMT': 'Withdrawal Amount',
            'DEPOSIT AMT': 'Deposit Amount',
            'BALANCE AMT': 'Balance Amount'
        }

        df.rename(columns=column_mapping, inplace=True)

        if 'Date' in df.columns:
            df['Original Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y', errors='coerce')
            df['Date Year'] = df['Original Date'].dt.year
            df['Date Month'] = df['Original Date'].dt.month
            df['Date Day'] = df['Original Date'].dt.day
            df.drop(columns=['Date'], inplace=True)

        if 'Value Date' in df.columns:
            df['Value Date'] = pd.to_datetime(df['Value Date'], format='%d-%b-%y', errors='coerce')
            df['Value Date Year'] = df['Value Date'].dt.year
            df['Value Date Month'] = df['Value Date'].dt.month
            df['Value Date Day'] = df['Value Date'].dt.day
            df.drop(columns=['Value Date'], inplace=True)

        def clean_numeric(value):
            if pd.isna(value):
                return np.nan
            value = str(value).replace(',', '').replace(' ', '').replace('₹', '')
            try:
                return float(value)
            except ValueError:
                return np.nan

        df['Withdrawal Amount'] = df['Withdrawal Amount'].apply(clean_numeric)
        df['Deposit Amount'] = df['Deposit Amount'].apply(clean_numeric)
        df['Balance Amount'] = df['Balance Amount'].apply(clean_numeric)

        df['Withdrawal Amount'] = df['Withdrawal Amount'].fillna(0)
        df['Deposit Amount'] = df['Deposit Amount'].fillna(0)
        df['Balance Amount'] = df['Balance Amount'].fillna(0)

        if 'Account No.' in df.columns:
            le = LabelEncoder()
            df['Account No.'] = le.fit_transform(df['Account No.'].astype(str))

        df['Transaction Count'] = df.groupby('Account No.')['Account No.'].transform('count')
        df['Transaction Frequency'] = df.groupby('Account No.')['Original Date'].transform(
            lambda x: x.diff().dt.days.fillna(0)).mean()
        df['Transaction Type'] = df['Transaction Details'].apply(
            lambda x: 1 if 'withdrawal' in x.lower() else (0 if 'deposit' in x.lower() else -1))
        df['Withdrawals Proportion'] = df.groupby('Account No.')['Transaction Type'].transform(
            lambda x: (x == 1).sum() / len(x))
        df['Deposits Proportion'] = df.groupby('Account No.')['Transaction Type'].transform(
            lambda x: (x == 0).sum() / len(x))

        features = [
            'Account No.', 'Withdrawal Amount', 'Deposit Amount', 'Balance Amount',
            'Transaction Count', 'Transaction Frequency', 'Withdrawals Proportion', 'Deposits Proportion'
        ]

        df_cleaned = df[features].copy()
        df_cleaned.dropna(inplace=True)

        return df_cleaned

    def detect_anomalies_with_tda_alternative(data_reduced):
        dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust parameters as needed
        clusters = dbscan.fit_predict(data_reduced)

        # Identify noise points (anomalies)
        anomalies = np.where(clusters == -1)[0]

        return anomalies

    def detect_anomalies_with_tda(self, data_reduced, graph):
        import numpy as np
        from sklearn.cluster import DBSCAN

        # Initialize an empty list to store indices of detected anomalies
        tda_anomalies = []

        # Print graph structure for debugging
        print("Graph Structure:", graph)

        # Calculate the degree of each node (i.e., number of connections)
        node_degrees = {node: len(edges) for node, edges in graph['nodes'].items() if isinstance(edges, list)}

        # Print node degrees for debugging
        print("Node Degrees:", node_degrees)

        # Extract degrees into a list and reshape for clustering
        degree_values = np.array(list(node_degrees.values()))[:, np.newaxis]

        # Use DBSCAN to identify anomalies based on node degrees
        dbscan = DBSCAN(eps=1.0, min_samples=2)  # Adjust parameters as needed
        clusters = dbscan.fit_predict(degree_values)

        # Print clusters for debugging
        print("DBSCAN Clusters:", clusters)

        # Anomalies are those points labeled as -1 (noise)
        anomalous_nodes = [node for node, cluster in zip(node_degrees.keys(), clusters) if cluster == -1]

        # Print anomalous nodes for debugging
        print("Anomalous Nodes:", anomalous_nodes)

        # Map anomalous nodes back to data points
        if anomalous_nodes:
            # Find indices corresponding to anomalous nodes
            node_to_index = {node: index for index, node in enumerate(graph['nodes'].keys())}
            tda_anomalies = [node_to_index[node] for node in anomalous_nodes if node in node_to_index]

        return tda_anomalies



    def plot_graphs(self, data_reduced, abs_z_scores, anomalies_mask_zscore, anomalies_mask_iso, df_cleaned):
        fig = go.Figure()

        # Normal Points
        fig.add_trace(go.Scatter3d(
            x=data_reduced[:, 0],
            y=data_reduced[:, 1],
            z=data_reduced[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color='blue',  # single color for all normal points
                opacity=0.5
            ),
            name='Normal Points'
        ))

        # Z-Score Anomalies
        fig.add_trace(go.Scatter3d(
            x=data_reduced[anomalies_mask_zscore, 0],
            y=data_reduced[anomalies_mask_zscore, 1],
            z=data_reduced[anomalies_mask_zscore, 2],
            mode='markers',
            marker=dict(
                size=7,
                color='red',  # single color for all Z-score anomalies
                opacity=0.8
            ),
            name='Z-Score Anomalies'
        ))

        # Isolation Forest Anomalies
        fig.add_trace(go.Scatter3d(
            x=data_reduced[anomalies_mask_iso, 0],
            y=data_reduced[anomalies_mask_iso, 1],
            z=data_reduced[anomalies_mask_iso, 2],
            mode='markers',
            marker=dict(
                size=7,
                color='green',  # single color for all Isolation Forest anomalies
                opacity=0.8
            ),
            name='Isolation Forest Anomalies'
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title='PCA1',
                yaxis_title='PCA2',
                zaxis_title='PCA3'
            ),
            title="Anomaly Detection in 3D PCA-reduced Space",
            legend=dict(x=0.1, y=0.9)
        )

        fig.show()

        # Additional Plots
        plt.figure(figsize=(10, 6))
        plt.hist(abs_z_scores.flatten(), bins=50, color='blue', alpha=0.7, label='Z-Scores')
        plt.axvline(x=2.2, color='red', linestyle='--', label='Threshold = 2.2')
        plt.title('Distribution of Z-Scores')
        plt.xlabel('Z-Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

        if not df_cleaned.empty:
            combined_anomalies = df_cleaned[df_cleaned['Combined_Anomaly'] == 1]
            plt.figure(figsize=(10, 6))
            plt.plot(df_cleaned.index, df_cleaned['Balance Amount'], label='Balance Amount', color='blue')
            plt.scatter(combined_anomalies.index, combined_anomalies['Balance Amount'], color='red', label='Anomalies')
            plt.title('Balance Amount Over Time with Anomalies')
            plt.xlabel('Transaction Index')
            plt.ylabel('Balance Amount')
            plt.legend()
            plt.show()

    def show_mapper_output(self):
        # Open the generated mapper_output.html file in the default web browser
        if self.file_name:
            webbrowser.open('mapper_output.html')
        else:
            messagebox.showerror("Error", "Mapper output not available.")

    def show_anomalies(self):
        # Check if anomalies data is available
        if self.anomalies_df is not None and not self.anomalies_df.empty:
            anomalies_window = tk.Toplevel(self)
            anomalies_window.title("Detected Anomalies")

            # Create a scrolled text area for displaying the anomalies data
            text_area = scrolledtext.ScrolledText(anomalies_window, wrap=tk.WORD, width=120, height=20)
            text_area.pack()

            # Filter the anomalies data to include only relevant columns
            relevant_columns = ['Withdrawal Amount', 'Deposit Amount', 'Balance Amount',
                                'Transaction Count', 'Transaction Frequency', 'ZScore_Anomaly',
                                'IsolationForest_Anomaly', 'Logistic_Anomaly', 'Combined_Anomaly']
            anomalies_df_display = self.anomalies_df[relevant_columns].copy()

            # Format the numerical values for better readability
            numerical_cols = ['Withdrawal Amount', 'Deposit Amount', 'Balance Amount',
                              'Transaction Count', 'Transaction Frequency']
            for col in numerical_cols:
                anomalies_df_display[col] = anomalies_df_display[col].round(2)

            # Optionally, add a row location (index) to display
            anomalies_df_display['Row Location'] = anomalies_df_display.index

            # Convert the filtered dataframe to a string with better formatting
            anomalies_text = anomalies_df_display.to_string(index=False, formatters={
                'Withdrawal Amount': '${:,.2f}'.format,
                'Deposit Amount': '${:,.2f}'.format,
                'Balance Amount': '${:,.2f}'.format,
                'Transaction Count': '{:,.0f}'.format,
                'Transaction Frequency': '{:,.2f}'.format,
                'ZScore_Anomaly': '{:.0f}'.format,
                'IsolationForest_Anomaly': '{:.0f}'.format,
                'Logistic_Anomaly': '{:.0f}'.format,
                'Combined_Anomaly': '{:.0f}'.format,
                'Row Location': '{:,.0f}'.format
            })

            # Insert the formatted text into the scrolled text area
            text_area.insert(tk.END, anomalies_text)
            text_area.configure(state=tk.DISABLED)
        else:
            messagebox.showerror("Error", "No anomalies detected or data not available.")


if __name__ == "__main__":
    app = AnomalyDetectionApp()
    app.mainloop()
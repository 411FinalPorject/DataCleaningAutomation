import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Load Dataset
def load_dataset(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)

# Step 2: Detect Anomalies - Statistical Methods
def detect_anomalies_statistical(df, column):
    """Detect anomalies in a single column using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return anomalies

# Step 3: Detect Anomalies - Machine Learning
def detect_anomalies_ml(df, feature_columns):
    """Detect anomalies across multiple columns using Isolation Forest."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_columns].select_dtypes(include=np.number))
    
    model = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly_flag'] = model.fit_predict(scaled_features)
    anomalies = df[df['anomaly_flag'] == -1]
    return anomalies

# Step 4: Data Cleaning
def clean_data(df, anomalies):
    """Remove anomalies from the dataset."""
    cleaned_data = df[~df.index.isin(anomalies.index)]
    return cleaned_data

# Step 5: Reporting
def generate_anomaly_report(anomalies):
    """Generate a report summarizing detected anomalies."""
    print("Anomaly Report:")
    print(anomalies.describe(include='all'))
    anomalies.to_csv("anomaly_report.csv", index=False)

# Step 6: Visualization
def visualize_anomalies(df, column, anomalies):
    """Visualize anomalies on a scatter plot for a single column."""
    plt.figure(figsize=(10, 6))
    plt.scatter(df.index, df[column], label='Data', color='blue', alpha=0.5)
    plt.scatter(anomalies.index, anomalies[column], label='Anomalies', color='red')
    plt.legend()
    plt.title(f"Anomalies in {column}")
    plt.show()

# Step 7: Comparison with Expected Results
def compare_results(cleaned_file, expected_file):
    """Compare the cleaned dataset with an expected results file."""
    cleaned_df = pd.read_csv(cleaned_file)
    expected_df = pd.read_csv(expected_file)
    
    comparison = cleaned_df.equals(expected_df)
    if comparison:
        print("The cleaned dataset matches the expected results.")
    else:
        print("The cleaned dataset does NOT match the expected results.")
        diff = cleaned_df.compare(expected_df, keep_equal=False)
        print("Differences:")
        print(diff)

# Main Execution
if __name__ == "__main__":
    # File path to dataset
    file_path = "generic_dummy_dataset.csv"

    # Load the dataset
    df = load_dataset(file_path)

    # Specify columns to analyze (automatic detection of numeric columns)
    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()

    for column in numerical_columns:
        # Detect anomalies using statistical methods
        anomalies_stat = detect_anomalies_statistical(df, column)
        print(f"Statistical anomalies in {column}:")
        print(anomalies_stat)

        # Detect anomalies using machine learning
        anomalies_ml = detect_anomalies_ml(df, numerical_columns)
        print(f"ML anomalies across all numerical columns:")
        print(anomalies_ml)

        # Clean the dataset
        df_cleaned = clean_data(df, anomalies_ml)

        # Generate reports and visualizations
        generate_anomaly_report(anomalies_ml)
        visualize_anomalies(df, column, anomalies_ml)

    # Save cleaned data
    cleaned_file = "cleaned_data.csv"
    df_cleaned.to_csv(cleaned_file, index=False)
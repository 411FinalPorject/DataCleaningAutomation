import pandas as pd
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(file_path):
    """
    Load CSV data with fixed dtype handling and better performance for mixed types.
    """
    logging.info(f"Loading data from {file_path}...")
    return pd.read_csv(file_path, low_memory=False)


def clean_data(df):
    """
    Clean data by converting columns with mixed types to numeric where applicable.
    """
    mixed_type_cols = ['term', 'int_rate', 'emp_length', 'revol_util', 'desc', 'zip_code', 'settlement_percentage']
    for col in mixed_type_cols:
        if col in df.columns:
            if df[col].dtype == 'object':  # Ensure column is a string before using .str
                # Handle percentage columns
                if col in ['int_rate', 'revol_util', 'settlement_percentage']:
                    df[col] = df[col].str.replace('%', '', regex=True).astype(float)
                    logging.info(f"Cleaned percentage format in column: {col}")
                elif col == 'term':
                    # Extract numeric values from "36 months", "60 months", etc.
                    df[col] = df[col].str.extract(r'(\d+)').astype(float)
                    logging.info(f"Extracted numeric term values in column: {col}")
                elif col == 'emp_length':
                    # Handle employment length (e.g., "10+ years", "<1 year")
                    df[col] = df[col].str.extract(r'(\d+)').astype(float)
                    logging.info(f"Extracted numeric employment length values in column: {col}")
                elif col == 'zip_code':
                    # Extract numeric parts of zip codes if necessary
                    df[col] = df[col].str.extract(r'(\d+)')
                    logging.info(f"Extracted numeric zip code values in column: {col}")
            else:
                logging.info(f"Column {col} is not an object; skipping string operations.")
    return df


def detect_statistical_anomalies(df, column):
    """
    Detect statistical anomalies using standard deviation.
    """
    if column not in df.columns:
        logging.warning(f"Column {column} not found in DataFrame!")
        return pd.DataFrame()

    mean = df[column].mean()
    std = df[column].std()
    anomalies = df[(df[column] > mean + 3 * std) | (df[column] < mean - 3 * std)]

    logging.info(f"Detected {len(anomalies)} statistical anomalies in {column}.")
    return anomalies


def detect_ml_anomalies(df, column):
    """
    Simulate ML anomaly detection by flagging rows with specific conditions.
    """
    if column not in df.columns:
        logging.warning(f"Column {column} not found in DataFrame!")
        return pd.DataFrame()

    anomalies = df[df[column] < 0]  # Example condition; replace with your ML model's output
    logging.info(f"Detected {len(anomalies)} ML anomalies in {column}.")
    return anomalies


def remove_anomalies(df, anomalies_list):
    """
    Remove anomalies from the DataFrame based on a combined anomaly index.
    """
    if anomalies_list:
        combined_anomalies = pd.concat(anomalies_list).drop_duplicates()
        cleaned_data = df.drop(index=combined_anomalies.index)
        logging.info(f"Removed {len(combined_anomalies)} anomalies from the dataset.")
        return cleaned_data, combined_anomalies
    return df, pd.DataFrame()


def plot_data(df, column):
    """
    Plot data with fixed legend location and improved readability.
    """
    if column not in df.columns:
        logging.warning(f"Column {column} not found in DataFrame!")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(df[column].dropna(), bins=50, alpha=0.7, label=column)
    plt.legend(loc="upper left")  # Explicit location for large datasets
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()


def save_to_csv(df, file_name):
    """
    Save a DataFrame to a CSV file.
    """
    logging.info(f"Saving data to {file_name}...")
    df.to_csv(file_name, index=False)


def main():
    file_path = "approved_data_2016_2018.csv"  # Replace with your actual file path
    cleaned_file = "cleaned_data.csv"
    anomaly_file = "anomalies_data.csv"

    # Load and clean the data
    df = load_data(file_path)
    df = clean_data(df)

    # Detect anomalies
    anomalies_list = []
    for column in ['loan_amnt', 'int_rate']:  # Replace with your actual columns
        stat_anomalies = detect_statistical_anomalies(df, column)
        ml_anomalies = detect_ml_anomalies(df, column)

        # Combine anomalies
        combined_anomalies = pd.concat([stat_anomalies, ml_anomalies]).drop_duplicates()
        anomalies_list.append(combined_anomalies)

        # Print anomaly summaries
        logging.info(f"Statistical anomalies in {column}:\n{stat_anomalies.head()}")
        logging.info(f"ML anomalies in {column}:\n{ml_anomalies.head()}")

    # Remove anomalies from the dataset
    df_cleaned, anomalies_df = remove_anomalies(df, anomalies_list)

    # Save cleaned data and anomalies to separate files
    save_to_csv(df_cleaned, cleaned_file)
    save_to_csv(anomalies_df, anomaly_file)

    # Plot data
    for column in ['loan_amnt', 'int_rate']:  # Replace with your actual columns
        plot_data(df_cleaned, column)


if __name__ == "__main__":
    main()

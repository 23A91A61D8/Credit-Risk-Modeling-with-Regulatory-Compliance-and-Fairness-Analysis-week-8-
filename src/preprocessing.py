import pandas as pd
import os


def load_data(path):
    """
    Load the raw dataset
    """
    df = pd.read_csv(path)

    # Remove unwanted index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    return df


def create_target(df):
    """
    Use existing numeric credit_risk column.
    0 = good loan
    1 = bad loan (default)
    """
    if 'credit_risk' not in df.columns:
        raise ValueError("Column 'credit_risk' not found in dataset!")

    # Since dataset already uses 0 and 1
    df['default'] = df['credit_risk']

    return df


def handle_missing_values(df):
    """
    German Credit dataset does not contain missing values.
    We verify and return dataframe.
    """
    total_missing = df.isnull().sum().sum()
    print("Total Missing Values in Dataset:", total_missing)

    return df


def check_class_balance(df):
    """
    Display class distribution
    """
    distribution = df['default'].value_counts(normalize=True)
    print("Class Distribution:")
    print(distribution)

    return distribution


def save_processed_data(df, output_path):
    """
    Save cleaned dataset
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
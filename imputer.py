import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

def impute_column_with_mice(csv_path: str, target_column: str) -> pd.DataFrame:
    """
    Imputes missing values in a specified column using MICE (IterativeImputer).
    
    Parameters:
        csv_path (str): Path to the input CSV file.
        target_column (str): The column to impute.
    
    Returns:
        pd.DataFrame: DataFrame with the imputed column.
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in dataset.")
    
    # Select numeric columns only (IterativeImputer works with numbers)
    numeric_df = df.select_dtypes(include=['number']).copy()
    
    if target_column not in numeric_df.columns:
        raise ValueError(f"Column '{target_column}' must be numeric for MICE imputation.")
    
    # Apply MICE (IterativeImputer)
    imputer = IterativeImputer(random_state=42)
    imputed_array = imputer.fit_transform(numeric_df)
    
    # Replace imputed column in original df
    numeric_imputed = pd.DataFrame(imputed_array, columns=numeric_df.columns)
    df[target_column] = numeric_imputed[target_column]
    
    return df

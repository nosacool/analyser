import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

def imputer(csv_path: str, target_column: str) -> pd.DataFrame:
    
    # Load data
    df = pd.read_csv(csv_path)
    
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in dataset.")
    
    df = df.select_dtypes(include=['category']).copy()
    print(df)
    
    imputer = IterativeImputer(random_state=42)
    imputed_array = imputer.fit_transform(df)
    
    numeric_imputed = pd.DataFrame(imputed_array, columns=df.columns)
    df[target_column] = numeric_imputed[target_column]
    
    return df

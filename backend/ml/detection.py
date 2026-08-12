import pandas as pd
import numpy as np

def detect_problem_type(series: pd.Series) -> tuple[str, str]:
    """
    Determines whether a target column requires Classification or Regression.
    
    Rules (from design specification):
    1. If target dtype == object or category or boolean -> Classification
    2. Else if target dtype is integer/numeric and unique_count <= 20 and (unique_count <= 2 or unique_count / total_rows <= 0.05) -> Classification
    3. Else -> Regression
    
    Returns:
        (problem_type, reasoning)
    """
    clean_series = series.dropna()
    total_rows = len(series)
    unique_count = clean_series.nunique()
    dtype_str = str(series.dtype).lower()
    
    if total_rows == 0 or unique_count == 0:
        raise ValueError("Target column contains no non-null values.")
        
    if unique_count == 1:
        raise ValueError("Target column has only 1 unique value. Model training requires at least 2 distinct target classes/values.")

    # Rule 1: Object or Category or Bool
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_bool_dtype(series):
        reasoning = (
            f"Target column '{series.name}' has non-numeric data type '{dtype_str}' "
            f"with {unique_count} unique categorical values across {total_rows} rows."
        )
        return "classification", reasoning

    # Rule 2: Integer / Discrete numeric with few unique values
    is_integer = pd.api.types.is_integer_dtype(series) or (
        pd.api.types.is_numeric_dtype(series) and (clean_series % 1 == 0).all()
    )
    
    ratio = unique_count / total_rows if total_rows > 0 else 1.0
    
    if is_integer and unique_count <= 20 and (unique_count <= 2 or ratio <= 0.05):
        reasoning = (
            f"Target column '{series.name}' has integer numeric type '{dtype_str}', "
            f"{unique_count} unique values (<= 20 threshold), and a unique-to-total ratio of "
            f"{ratio:.4f} (<= 0.05 threshold or binary target <= 2 classes), indicating discrete classification labels."
        )
        return "classification", reasoning

    # Rule 3: Continuous numeric -> Regression
    reasoning = (
        f"Target column '{series.name}' has continuous numeric type '{dtype_str}' "
        f"with {unique_count} distinct values across {total_rows} rows (ratio {ratio:.4f}), "
        f"indicating a continuous regression task."
    )
    return "regression", reasoning

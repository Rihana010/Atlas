import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    """
    Leak-free dataset preprocessor.
    Fits imputers, encoders, and scalers strictly on the training set,
    then transforms both training and test sets.
    """
    def __init__(self, problem_type: str, target_column: str):
        self.problem_type = problem_type
        self.target_column = target_column
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.feature_names: List[str] = []
        self.target_encoder: Any = None
        self.preprocessor: Any = None
        self.scaler: Any = None

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Processes dataframe into train/test splits.
        Returns:
            X_train, X_test (unscaled, imputed+encoded)
            X_train_scaled, X_test_scaled (scaled, for linear models)
            y_train, y_test
            metadata dict
        """
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")

        # Drop rows where target is NaN
        df_clean = df.dropna(subset=[self.target_column]).copy()
        if len(df_clean) == 0:
            raise ValueError(f"Target column '{self.target_column}' has no valid non-null rows.")

        X = df_clean.drop(columns=[self.target_column])
        y = df_clean[self.target_column]

        # Identify feature column types
        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        if len(self.num_cols) == 0 and len(self.cat_cols) == 0:
            raise ValueError("Dataset has no usable feature columns after target removal.")

        # Encode target for classification if non-numeric or string
        if self.problem_type == "classification":
            class_counts = y.value_counts()
            if len(class_counts) < 2:
                raise ValueError("Classification target must have at least 2 distinct classes.")
            
            # Robust check for non-numeric target (handles string, object, category, bool)
            if not pd.api.types.is_numeric_dtype(y) or y.dtype == object or str(y.dtype).startswith('cat') or str(y.dtype).startswith('str'):
                self.target_encoder = LabelEncoder()
                y_encoded = self.target_encoder.fit_transform(y.astype(str))
            else:
                y_encoded = y.to_numpy()
                
            # Stratified split if min class count >= 2
            stratify_y = y_encoded if class_counts.min() >= 2 else None
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=stratify_y
            )
        else:
            # Regression target
            if not pd.api.types.is_numeric_dtype(y):
                raise ValueError(f"Target column '{self.target_column}' contains non-numeric string values and cannot be used for regression.")
            y_encoded = y.to_numpy().astype(np.float64)
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=None
            )

        # Build feature transformers
        transformers = []
        if self.num_cols:
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median'))
            ])
            transformers.append(('num', num_pipeline, self.num_cols))
            
        if self.cat_cols:
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', cat_pipeline, self.cat_cols))

        self.preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

        # FIT PREPROCESSOR STRICTLY ON X_train_raw (LEAK FREE)
        X_train_proc = self.preprocessor.fit_transform(X_train_raw)
        X_test_proc = self.preprocessor.transform(X_test_raw)

        # FIT SCALER STRICTLY ON X_train_proc FOR LINEAR MODELS (LEAK FREE)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_proc)
        X_test_scaled = self.scaler.transform(X_test_proc)

        # Extract generated feature names
        feature_names = []
        if self.num_cols:
            feature_names.extend(self.num_cols)
        if self.cat_cols:
            try:
                cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['encoder']
                encoded_cat_names = cat_encoder.get_feature_names_out(self.cat_cols).tolist()
                feature_names.extend(encoded_cat_names)
            except Exception:
                feature_names.extend([f"cat_{i}" for i in range(X_train_proc.shape[1] - len(self.num_cols))])

        self.feature_names = feature_names

        meta = {
            "num_features_count": len(self.num_cols),
            "cat_features_count": len(self.cat_cols),
            "processed_feature_count": X_train_proc.shape[1],
            "train_samples": X_train_proc.shape[0],
            "test_samples": X_test_proc.shape[0],
        }

        return X_train_proc, X_test_proc, X_train_scaled, X_test_scaled, y_train, y_test, meta

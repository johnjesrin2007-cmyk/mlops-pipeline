import pandas as pd
from typing import Tuple, Optional


def preprocess_data(
    path: str,
    training: bool = True,
    target_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Preprocess data for training or inference.

    Args:
        path: CSV file path
        training: True for training, False for inference
        target_col: target column name (required for training)

    Returns:
        X: Features
        y: Target (None during inference)
    """

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty")

    if training:
        if target_col is None:
            target_col = df.columns[-1]  # safe default

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y

    else:
        # inference mode
        return df, None


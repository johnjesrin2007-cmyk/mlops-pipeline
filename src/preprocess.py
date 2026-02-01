import pandas as pd
from prefect import task

@task
def preprocess_data(path):
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty")

    print("Columns found:", df.columns)

    target_col = df.columns[-1]   # last column as target (SAFE)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y

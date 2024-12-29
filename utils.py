import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def fill_missing_value(df, object_columns, float_columns):
    df[object_columns] = df[object_columns].fillna(method="pad")
    df[float_columns] = df[float_columns].apply(lambda col: col.fillna(col.mean()))
    return df

def encode_categorical_columns(df, object_columns):
    encoder = LabelEncoder()
    for column in object_columns:
        df[column] = encoder.fit_transform(df[column])
    return df

def scale_features(X, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    return X, scaler
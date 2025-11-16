import re
import pandas as pd
import numpy as np
from typing import Tuple
from . import config


def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Clean HVAC Operation Mode: lower + strip spaces
    if "HVAC Operation Mode" in df.columns:
        df["HVAC Operation Mode"] = (
            df["HVAC Operation Mode"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

    # Clean Activity Level naming: add underscore between camelCase, replace spaces with underscore, lower
    if config.TARGET_COL in df.columns:
        col = config.TARGET_COL
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace(r"([a-z])([A-Z])", r"\1_\2", regex=True)
        df[col] = df[col].str.replace(" ", "_")
        df[col] = df[col].str.lower()

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Numeric columns with missing values
    if "CO2_ElectroChemicalSensor" in df.columns:
        df["CO2_ElectroChemicalSensor"] = df["CO2_ElectroChemicalSensor"].fillna(
            df["CO2_ElectroChemicalSensor"].median()
        )

    if "MetalOxideSensor_Unit3" in df.columns:
        df["MetalOxideSensor_Unit3"] = df["MetalOxideSensor_Unit3"].fillna(
            df["MetalOxideSensor_Unit3"].median()
        )

    # Categorical with missing values
    for col in ["CO_GasSensor", "Ambient Light Level"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def clip_unrealistic_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Temperature" in df.columns:
        df["Temperature"] = df["Temperature"].clip(lower=10, upper=50)

    if "Humidity" in df.columns:
        df["Humidity"] = df["Humidity"].clip(lower=0, upper=100)

    # CO2 readings: clip negatives to 0
    if "CO2_InfraredSensor" in df.columns:
        df["CO2_InfraredSensor"] = df["CO2_InfraredSensor"].clip(lower=0)

    if "CO2_ElectroChemicalSensor" in df.columns:
        df["CO2_ElectroChemicalSensor"] = df["CO2_ElectroChemicalSensor"].clip(lower=0)

    return df


def prepare_features(include_metaloxide3: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Full preprocessing pipeline:
    - load raw data
    - drop duplicates
    - clean categorical naming
    - handle missing values
    - clip unrealistic values
    - drop Session ID
    - optionally drop MetalOxideSensor_Unit3
    - one-hot encode categorical features
    """
    from .data_loader import load_data  # local import to avoid circular imports

    df = load_data()

    # Drop exact duplicates (as done in EDA)
    df = df.drop_duplicates().copy()

    # Clean categorical columns (HVAC, Activity Level)
    df = clean_categorical_columns(df)

    # Handle missing values
    df = handle_missing_values(df)

    # Clip unrealistic values
    df = clip_unrealistic_values(df)

    # Drop Session ID (identifier, not feature)
    if "Session ID" in df.columns:
        df = df.drop(columns=["Session ID"])

    # Optionally drop MetalOxideSensor_Unit3 for alternative pipeline
    if not include_metaloxide3 and "MetalOxideSensor_Unit3" in df.columns:
        df = df.drop(columns=["MetalOxideSensor_Unit3"])

    # Separate target
    y = df[config.TARGET_COL]
    X = df.drop(columns=[config.TARGET_COL])

    # One-hot encode categorical features using pandas
    X_encoded = pd.get_dummies(X, drop_first=False)

    return X_encoded, y
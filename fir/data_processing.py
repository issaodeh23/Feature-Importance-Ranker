import click
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
from .logger_setup import setup_logger

logger = setup_logger("fir_log.txt", to_console=True)


def load_data(
    filepath: str,
    target_column: str,
    columns_to_drop: tuple = (),
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads a parquet or csv file, performs data cleaning, and returns the features and target variable.

    The function reads a parquet/CSV file from the specified path, replaces common NA values with np.nan,
    drops specified columns and any rows containing missing values, removes constant columns
    (except for the target), and converts all columns to numeric types. If any column cannot be
    fully converted to numeric, an error is raised. The cleaned feature matrix and target vector
    are then returned.

    :param filepath: Path to the parquet/CSV file to load.
    :param target_column: Name of the column to use as the target variable.
    :param columns_to_drop: Tuple of column names to drop from the data (default: ()).

    :raises KeyError: If the target column is missing after cleaning.
    :raises ValueError: If the DataFrame is empty after cleaning, or if any column cannot be fully converted to numeric.

    :return: Tuple (X, y) where X is the cleaned feature DataFrame and y is the target Series.
    """
    # Detect file type and load accordingly
    if filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
        logger.info("Loaded parquet file.")
    elif filepath.endswith(".csv"):
        df = pd.read_csv(filepath, low_memory=False)
        logger.info("Loaded CSV file.")
    else:
        logger.error("Unsupported file type. Please provide a .parquet or .csv file.")
        raise ValueError(
            "Unsupported file type. Please provide a .parquet or .csv file."
        )

    # replaces all NA values with pandas NaN
    df.replace(
        to_replace=["N/A", "NA", "na", "NaN", "nan", "null", "NULL", "\\N", "", "//n"],
        value=np.nan,
        inplace=True,
    )
    # drop the specified columns if they exist
    if columns_to_drop is not None:
        df.drop(columns=list(columns_to_drop), inplace=True)
    logger.info(f"Dropping specified columns: {columns_to_drop}")

    df.dropna(axis=0, how="any", inplace=True)  # drops any row that has NaN in it.
    df.reset_index(
        drop=True, inplace=True
    )  # resets indeces after dropping all the rows and prevents pandas from adding a new columns of the old indices
    logger.info("Dropped rows with NaN values.")

    # checks if any column value has is constant accross all the rows if so removes it
    constant_cols = [
        col
        for col in df.columns
        if df[col].nunique(dropna=False) == 1 and col != target_column
    ]
    df = df.drop(columns=constant_cols)
    logger.info(f"Dropping constant columns: {constant_cols}")

    # raises KeyError if the target column is not in the dataframe
    if target_column not in df.columns:
        logger.error(
            f"The target column {target_column}, does not exist in the dataframe after cleaning."
        )
        raise KeyError(
            f"The target column {target_column}, does not exist in the dataframe after cleaning."
        )

    if df.empty:
        logger.error("The DataFrame is empty after cleaning. Please check your data.")
        raise ValueError(
            "The DataFrame is empty after cleaning. Please check your data."
        )

    # Convert all columns to numeric, raising an error if any column cannot be fully converted
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isnull().any():
            logger.error(f"Column '{col}' cannot be fully converted to numeric.")
            raise ValueError(f"Column '{col}' cannot be fully converted to numeric.")
        df[col] = converted

    y = df[target_column].copy()
    X = df.drop(columns=[target_column])

    return X, y


def split_data(
    data_tuple,
    train_size: float,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the cleaned data into training and testing sets using sklearn's train_test_split.

    The function divides the provided feature and target data into training and testing subsets,
    stratifying the split if the target variable contains at least two instances of each class.
    The split proportions and random seed are controlled by the function arguments.

    :param data_tuple: Tuple (X, y) where X is the feature DataFrame and y is the target Series.
    :param train_size: Proportion of the data to include in the training set.
    :param test_size: Proportion of the data to include in the test set.
    :param random_state: Random seed for reproducibility.

    :return: Tuple (X_train, X_test, y_train, y_test) with the split data.
    """
    data_y = data_tuple[1]
    counts = data_y.value_counts()

    if len(counts) > 1 and counts.min() >= 2:
        stratify_arg = data_y
    else:
        stratify_arg = None

    X_train, X_test, y_train, y_test = train_test_split(
        *data_tuple,
        train_size=train_size,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_arg,
    )

    logger.info("✔ Data split into training and testing sets successfully.")
    logger.info(f"Shape of X_train: {X_train.shape}, Shape of y_train: {y_train.shape}")
    logger.info(f"Shape of X_test: {X_test.shape}, Shape of y_test: {y_test.shape}")
    logger.info(
        f"Train size: {train_size}, Test size: {test_size}, Random state: {random_state}"
    )

    return X_train, X_test, y_train, y_test

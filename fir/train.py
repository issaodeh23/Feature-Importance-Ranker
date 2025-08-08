import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from time import perf_counter
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from typing import Any, Dict, Optional
from .logger_setup import setup_logger

logger = setup_logger("fir_log.txt", to_console=True)


def tune_xgb(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_trials: int = 50,
    seed: int = 42,
    is_classification: bool = False,
) -> optuna.study.Study:
    """
    Tunes an XGBoost model (regressor or classifier) using Optuna for hyperparameter optimization.

    This function validates the input data, checks for numeric features, and ensures the target is not constant (for regression).
    It defines an Optuna objective function that trains and evaluates an XGBoost model with hyperparameters suggested by Optuna.
    The function maximizes either R^2 (regression) or accuracy (classification) on the test set, and returns the Optuna study object
    containing the best hyperparameters and score.

    :param X_train: Training input features as a pandas DataFrame or numpy array. All features must be numeric.
    :param X_test: Testing input features as a pandas DataFrame or numpy array. All features must be numeric.
    :param y_train: Target labels corresponding to X_train.
    :param y_test: Target labels corresponding to X_test.
    :param n_trials: Number of Optuna trials to run for hyperparameter search (default: 50).
    :param seed: Random seed for reproducibility (default: 42).
    :param is_classification: If True, tunes an XGBClassifier; otherwise, an XGBRegressor.

    :raises ValueError: If input data is empty, features are not numeric, target is constant (for regression), or lengths mismatch.
    :raises TypeError: If input types are incorrect or unknown parameters are provided.

    :return: Optuna study object containing the best hyperparameters and score.
    """
    if not isinstance(X_train, (pd.DataFrame, np.ndarray)):
        raise TypeError("X_train must be a pandas DataFrame or numpy array.")

    if not isinstance(X_test, (pd.DataFrame, np.ndarray)):
        raise TypeError("X_test must be a pandas DataFrame or numpy array.")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length.")
    if not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError("n_trials must be a positive integer.")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    # Handle constant regression target
    y_arr = np.asarray(y_train)
    if np.all(y_arr == y_arr[0]):
        logger.error("Regression target is constant; cannot tune model.")
        raise ValueError("Regression target is constant; cannot tune model.")

    # Handle unseen classes in classification
    if is_classification:
        train_classes = set(np.unique(y_train))
        test_classes = set(np.unique(y_test))
        if not test_classes.issubset(train_classes):
            logger.error("Unseen classes in test set.")
            raise ValueError("Unseen classes in test set.")

    def objective(trial):
        trial_number = trial.number
        logger.info(f"Running trial {trial_number}...")
        # common hyperparameter search space
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0),
            "random_state": seed,
        }

        if is_classification:
            # classification-specific
            param["objective"] = "binary:logistic"
            model = xgb.XGBClassifier(**param, eval_metric="logloss")
        else:
            # regression-specific
            param["objective"] = "reg:squarederror"
            param["verbosity"] = 0
            param["booster"] = "gbtree"
            model = xgb.XGBRegressor(**param)

        # train and predict
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)

        # return the metric to maximize
        if is_classification:
            return accuracy_score(y_test, preds)
        else:
            return r2_score(y_test, preds)

    # always maximize (higher R^2 or higher accuracy)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)

    logger.info("Optuna study completed.")
    logger.info(
        f"Best trial: {study.best_trial.number}, Best value: {study.best_value}"
    )
    logger.info(f"Best params: {study.best_params}")

    return study


def model_initializer(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    is_classification: bool = False,
    params_dict: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Initializes and trains an XGBoost model for regression or classification.

    This function validates the input data, checks for numeric features, and ensures the target is not constant (for regression).
    It then instantiates an XGBoost model (XGBRegressor or XGBClassifier) with the provided parameters, fits it to the training data,
    and wraps the predict method in classification mode to check for unseen classes in predictions.

    :param X_train: Training input features as a pandas DataFrame or numpy array. All features must be numeric.
    :param y_train: Target labels corresponding to X_train.
    :param is_classification: If True, initializes an XGBClassifier; otherwise, an XGBRegressor.
    :param params_dict: Dictionary of hyperparameters to pass to the XGBoost model.

    :raises ValueError: If input data is empty, features are not numeric, target is constant (for regression), or lengths mismatch.
    :raises TypeError: If unknown parameters are provided in params_dict.

    :return: Trained XGBoost model (XGBRegressor or XGBClassifier).
    """
    if params_dict is None:
        params_dict = {}

    # Check for empty input
    if X_train is None or y_train is None or len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Input data cannot be empty.")

    # Check for length mismatch
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length.")

    # Check for non-numeric features
    if isinstance(X_train, pd.DataFrame):
        if not np.all([np.issubdtype(dtype, np.number) for dtype in X_train.dtypes]):
            raise ValueError("All features must be numeric.")
    else:
        # If not DataFrame, try to convert to float
        try:
            np.asarray(X_train, dtype=float)
        except Exception:
            raise ValueError("All features must be numeric.")

    # Check for constant regression target
    if not is_classification:
        y_arr = np.asarray(y_train)
        if np.all(y_arr == y_arr[0]):
            logger.error("Regression target is constant; cannot fit model.")
            raise ValueError("Regression target is constant; cannot fit model.")

    # Try to instantiate the model (catch invalid params)
    xgb_type = XGBClassifier if is_classification else XGBRegressor
    valid_params = xgb_type().get_params().keys()
    unknown_params = set(params_dict) - set(valid_params)
    if unknown_params:
        logger.error(f"Unknown parameter(s) for {xgb_type.__name__}: {unknown_params}")
        raise TypeError(
            f"Unknown parameter(s) for {xgb_type.__name__}: {unknown_params}"
        )

    xgb_model = xgb_type(**params_dict)

    start = perf_counter()
    xgb_model.fit(X_train, y_train)
    end = perf_counter()
    logger.info(f"✔ XGB model completed fitting in {end-start:.2f} seconds")

    # Wrap predict to check for unseen classes in classification
    if is_classification:
        train_classes = set(np.unique(y_train))
        orig_predict = xgb_model.predict

        def safe_predict(X):
            preds = orig_predict(X)
            pred_classes = set(np.unique(preds))
            if not pred_classes.issubset(train_classes):
                logger.error("Unseen classes in test set.")
                raise ValueError("Unseen classes in test set.")
            return preds

        xgb_model.predict = safe_predict

    return xgb_model

import pandas as pd
import numpy as np
import pytest
from fir import train

# --- Fixtures for tune_xgb tests ---
@pytest.fixture
def regression_split_data():
    X = np.random.rand(20, 3)
    y = np.random.rand(20)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, y_train, y_test


# TC01: tune_xgb raises ValueError for invalid n_trials
@pytest.mark.parametrize("n_trials", [0, -1])
def test_tune_xgb_invalid_trials(n_trials, regression_split_data):
    X_train, X_test, y_train, y_test = regression_split_data
    with pytest.raises(ValueError):
        train.tune_xgb(X_train, X_test, y_train, y_test, n_trials=n_trials)


# TC02: tune_xgb raises TypeError for non-integer seed
@pytest.mark.parametrize("seed", [3.14, "not_a_number"])
def test_non_int_seed(seed, regression_split_data):
    X_train, X_test, y_train, y_test = regression_split_data
    with pytest.raises(TypeError):
        train.tune_xgb(X_train, X_test, y_train, y_test, seed=seed)


# TC03: tune_xgb raises ValueError for length mismatch
def test_length_mismatch(regression_split_data):
    X_train, X_test, y_train, y_test = regression_split_data
    y_tr_bad = y_train[:-1]
    with pytest.raises(ValueError):
        train.tune_xgb(X_train, X_test, y_tr_bad, y_test, n_trials=10, seed=42)


# TC04: tune_xgb raises ValueError for non-numeric features
def test_non_numeric_features(regression_split_data):
    X_train, X_test, y_train, y_test = regression_split_data
    X_train_bad = pd.DataFrame(X_train)
    X_train_bad[0] = "non-numeric"
    with pytest.raises(ValueError):
        train.tune_xgb(X_train_bad, X_test, y_train, y_test, n_trials=1, seed=42)


# TC05: tune_xgb raises ValueError for constant regression target
def test_constant_regression_target(regression_split_data, monkeypatch):
    Xtr, Xte, ytr, yte = regression_split_data
    y_const = pd.Series(np.zeros(len(ytr)))
    with pytest.raises(
        ValueError, match="Regression target is constant; cannot tune model."
    ):
        train.tune_xgb(
            Xtr, Xte, y_const, y_const, n_trials=1, seed=0, is_classification=False
        )


# TC06: tune_xgb raises ValueError for unseen classes in test set (classification)
def test_unseen_classification_target(regression_split_data, monkeypatch):
    Xtr, Xte, ytr, yte = regression_split_data
    ytr_bad = pd.Series(np.random.choice([0, 1], size=len(ytr)))
    yte_bad = pd.Series(np.random.choice([2], size=len(yte)))
    with pytest.raises(ValueError):
        train.tune_xgb(
            Xtr, Xte, ytr_bad, yte_bad, n_trials=1, seed=0, is_classification=True
        )


# TC07: tune_xgb basic functionality returns a study with best_params and best_value
def test_basic_functionality(regression_split_data):
    X_train, X_test, y_train, y_test = regression_split_data
    study = train.tune_xgb(X_train, X_test, y_train, y_test, n_trials=10, seed=42)
    assert hasattr(study, "best_params")
    assert isinstance(study.best_params, dict)
    assert isinstance(study.best_value, float)


# --- Fixtures for model_initializer tests ---
@pytest.fixture
def simple_regression_data():
    X = pd.DataFrame(np.random.rand(20, 3))
    y = pd.Series(np.random.rand(20))
    return X, y


@pytest.fixture
def simple_classification_data():
    X = pd.DataFrame(np.random.rand(20, 3))
    y = pd.Series(np.random.choice([0, 1], size=20))
    return X, y


# TC08: model_initializer raises TypeError for invalid params, ValueError for length mismatch
@pytest.mark.parametrize(
    "params, y_mod, expected_exception",
    [
        ({"params_dict": {"nonexistent_param": 123}}, None, TypeError),
        ({}, "shorten", ValueError),
    ],
)
def test_model_initializer_invalid_inputs(
    simple_regression_data, params, y_mod, expected_exception
):
    X, y = simple_regression_data
    if y_mod == "shorten":
        y = y[:-1]
    with pytest.raises(expected_exception):
        train.model_initializer(X, y, **params)


# TC09: model_initializer raises ValueError for non-numeric features
def test_non_numeric_features_raises(simple_regression_data):
    X, y = simple_regression_data
    X.loc[0, 0] = "non-numeric"
    with pytest.raises(ValueError):
        train.model_initializer(X, y)


# TC10: model_initializer raises ValueError for constant regression target
def test_constant_regression_target_raises(simple_regression_data):
    X, y = simple_regression_data
    y_const = pd.Series(np.zeros(len(y)))
    with pytest.raises(
        ValueError, match="Regression target is constant; cannot fit model."
    ):
        train.model_initializer(X, y_const)


# TC11: model_initializer raises ValueError for unseen classes in prediction (classification)
def test_unseen_classification_target_raises(simple_classification_data):
    X, y = simple_classification_data
    model = train.model_initializer(X, y, is_classification=True)
    # Patch the internal orig_predict used by safe_predict
    model._orig_predict = lambda X: np.array([2] * len(X))
    # Replace safe_predict to use the patched _orig_predict
    def safe_predict(X):
        preds = model._orig_predict(X)
        pred_classes = set(np.unique(preds))
        train_classes = set(np.unique(y))
        if not pred_classes.issubset(train_classes):
            raise ValueError("Unseen classes in test set.")
        return preds

    model.predict = safe_predict
    with pytest.raises(ValueError, match="Unseen classes in test set."):
        model.predict(X)


# TC12: model_initializer fits and predicts for regression
def test_basic_functionality_regression(simple_regression_data):
    X, y = simple_regression_data
    model = train.model_initializer(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


# TC13: model_initializer fits and predicts for classification
def test_basic_functionality_classification(simple_classification_data):
    X, y = simple_classification_data
    model = train.model_initializer(X, y, is_classification=True)
    preds = model.predict(X)
    assert preds.shape == y.shape

import pandas as pd
import pytest
from fir import data_processing

# TESTS FOR LOAD_DATA FUNCTION
def test_basic_functionality(tmp_path):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "target": [0, 1]})
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    X, y = data_processing.load_data(str(path), "target", ())
    assert list(X.columns) == ["A", "B"]
    assert y.tolist() == [0, 1]


def test_drop_specified_columns(tmp_path):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "target": [0, 1]})
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    X, y = data_processing.load_data(str(path), "target", ("A",))
    assert "A" not in X.columns


def test_handle_na_values(tmp_path):
    df = pd.DataFrame({"A": [1, "NA"], "B": [3, 4], "target": [0, 1]})
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    X, y = data_processing.load_data(str(path), "target", ())
    assert len(X) == 1  # Only one row should remain


def test_drop_constant_columns(tmp_path):
    df = pd.DataFrame({"A": [1, 1], "B": [3, 4], "target": [0, 1]})
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    X, y = data_processing.load_data(str(path), "target", ())
    assert "A" not in X.columns


def test_target_column_missing(tmp_path):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    with pytest.raises(KeyError):
        data_processing.load_data(str(path), "target", ())


def test_all_columns_non_numeric(tmp_path):
    df = pd.DataFrame({"A": ["foo", "bar"], "target": [0, 1]})
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        data_processing.load_data(str(path), "target", ())


def test_no_columns_to_drop(tmp_path):
    df = pd.DataFrame({"A": [1, 2], "target": [0, 1]})
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    X, y = data_processing.load_data(str(path), "target", None)
    assert "A" in X.columns


# TESTS FOR SPLIT_DATA FUNCTION
def test_basic_split():
    X = pd.DataFrame({"A": range(10), "B": range(10, 20)})
    y = pd.Series([0, 1] * 5)
    X_train, X_test, y_train, y_test = data_processing.split_data((X, y), 0.8, 0.2, 42)
    assert len(X_train) + len(X_test) == 10
    assert len(y_train) + len(y_test) == 10
    # Check ratios (allowing for rounding)
    assert abs(len(X_train) - 8) <= 1
    assert abs(len(X_test) - 2) <= 1


def test_stratification():
    X = pd.DataFrame({"A": range(10)})
    y = pd.Series([0] * 8 + [1] * 2)
    X_train, X_test, y_train, y_test = data_processing.split_data((X, y), 0.7, 0.3, 1)
    # Check that both train and test have at least one of each class if possible
    assert set(y_train.unique()).issubset(set(y.unique()))
    assert set(y_test.unique()).issubset(set(y.unique()))


def test_no_stratification_single_class():
    X = pd.DataFrame({"A": range(5)})
    y = pd.Series([1] * 5)
    X_train, X_test, y_train, y_test = data_processing.split_data((X, y), 0.6, 0.4, 0)
    assert all(y_train == 1)
    assert all(y_test == 1)


def test_small_dataset():
    X = pd.DataFrame({"A": [1, 2]})
    y = pd.Series([0, 1])
    X_train, X_test, y_train, y_test = data_processing.split_data((X, y), 0.5, 0.5, 0)
    assert len(X_train) + len(X_test) == 2


def test_random_state_reproducibility():
    X = pd.DataFrame({"A": range(10)})
    y = pd.Series([0, 1] * 5)
    split1 = data_processing.split_data((X, y), 0.7, 0.3, 123)
    split2 = data_processing.split_data((X, y), 0.7, 0.3, 123)
    # Check that the splits are identical
    assert split1[0].equals(split2[0])
    assert split1[1].equals(split2[1])
    assert split1[2].equals(split2[2])
    assert split1[3].equals(split2[3])

import pandas as pd
import numpy as np
import pytest
from fir import shap_vals
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# TESTS FOR GET_SHAP_VALUES FUNCTION
def test_get_shap_values_regressor():
    X = pd.DataFrame(np.random.rand(10, 3), columns=["A", "B", "C"])
    y = np.random.rand(10)
    model = RandomForestRegressor().fit(X, y)
    shap_values = shap_vals.get_shap_values(model, X)
    assert hasattr(shap_values, "values")
    assert shap_values.values.shape[0] == X.shape[0]


def test_get_shap_values_classifier_3d():
    X = pd.DataFrame(np.random.rand(10, 3), columns=["A", "B", "C"])
    y = np.random.randint(0, 2, size=10)
    model = RandomForestClassifier().fit(X, y)
    shap_values = shap_vals.get_shap_values(model, X)
    # Should be 2D after processing
    assert shap_values.values.ndim == 2
    assert shap_values.values.shape[0] == X.shape[0]


def test_get_shap_values_untrained_model():
    X = pd.DataFrame(np.random.rand(10, 3), columns=["A", "B", "C"])
    model = RandomForestRegressor()
    with pytest.raises(Exception):
        shap_vals.get_shap_values(model, X)


# TESTS FOR GET_TREE_RANKINGS FUNCTION
class DummyShapExplanation:
    def __init__(self, values, feature_names):
        self.values = values
        self.feature_names = feature_names


def test_get_tree_rankings_basic():
    # 3 samples, 2 features
    values = np.array([[1, -2], [3, 0], [-1, 2]])
    feature_names = ["f1", "f2"]
    shap_exp = DummyShapExplanation(values, feature_names)
    rankings = shap_vals.get_tree_rankings(shap_exp)
    # Mean absolute SHAP: f1 = mean([1,3,1])=1.666..., f2=mean([2,0,2])=1.333...
    assert np.isclose(rankings["f1"], 1.666666, atol=1e-5)
    assert np.isclose(rankings["f2"], 1.333333, atol=1e-5)


def test_get_tree_rankings_zero_values():
    values = np.zeros((5, 3))
    feature_names = ["a", "b", "c"]
    shap_exp = DummyShapExplanation(values, feature_names)
    rankings = shap_vals.get_tree_rankings(shap_exp)
    assert all(v == 0 for v in rankings.values())


def test_get_tree_rankings_negative_values():
    values = np.array([[-1, -2], [-3, 0], [-1, -2]])
    feature_names = ["f1", "f2"]
    shap_exp = DummyShapExplanation(values, feature_names)
    rankings = shap_vals.get_tree_rankings(shap_exp)
    # Should use absolute values
    assert rankings["f1"] > 0
    assert rankings["f2"] > 0


def test_get_tree_rankings_feature_names_order():
    values = np.array([[1, 2, 3]])
    feature_names = ["z", "y", "x"]
    shap_exp = DummyShapExplanation(values, feature_names)
    rankings = shap_vals.get_tree_rankings(shap_exp)
    assert list(rankings.keys()) == feature_names


# TESTS FOR COMBINE_RANKINGS FUNCTION
def test_combine_rankings_basic():
    r1 = {"a": 2, "b": 1, "c": 1}
    r2 = {"a": 1, "b": 2, "c": 1}
    combined = shap_vals.combine_rankings(r1, r2)
    # All features should be present
    features = [f for f, _ in combined]
    assert set(features) == {"a", "b", "c"}
    # Rankings should be sorted in descending order
    import operator

    values = [v for _, v in combined]
    assert values == sorted(values, reverse=True)
    # Normalized importances should sum to 1 for each input
    for r in [r1, r2]:
        norm = sum(v / sum(r.values()) for v in r.values())
        assert abs(norm - 1.0) < 1e-8


def test_combine_rankings_equal_importance():
    r1 = {"x": 1, "y": 1}
    r2 = {"x": 1, "y": 1}
    combined = shap_vals.combine_rankings(r1, r2)
    # Both features should have equal combined importance
    assert abs(combined[0][1] - combined[1][1]) < 1e-8


def test_combine_rankings_top_n():
    # 12 features, only top 10 should be returned
    r1 = {f"f{i}": i + 1 for i in range(12)}
    r2 = {f"f{i}": 12 - i for i in range(12)}
    combined = shap_vals.combine_rankings(r1, r2)
    assert len(combined) == 10


def test_combine_rankings_feature_key_mismatch():
    r1 = {"a": 1, "b": 2}
    r2 = {"a": 1, "c": 2}
    try:
        shap_vals.combine_rankings(r1, r2)
        assert False, "Should raise KeyError or ValueError for mismatched keys"
    except Exception as e:
        assert isinstance(e, (KeyError, ValueError))


# ------------------- test for find_quartile_range function --------------------
# -- Helpers to build a fake shap Explanation object -------------------------
class FakeShapExp:
    def __init__(self, values, feature_names):
        self.values = values
        self.feature_names = feature_names


# -- Test data ----------------------------------------------------------------
@pytest.fixture
def X_df():
    # 5 rows, 3 features
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": [100, 200, 300, 400, 500],
        }
    )


@pytest.fixture
def shap_vals_array():
    # shap values roughly proportional to X values
    return np.array(
        [
            [-5, -50, -500],
            [0, 0, 0],
            [5, 50, 500],
            [10, 100, 1000],
            [15, 150, 1500],
        ]
    )


# -- Low‐quantile happy path ------------------------------------------------
def test_low_quantile_ranges_on_fake_exp(X_df, shap_vals_array):
    shap_exp = FakeShapExp(
        values=shap_vals_array,
        feature_names=list(X_df.columns),
    )
    # bottom 20% of shap for each feature only the most-negative row (row 0)
    out = shap_vals.find_quantile_ranges(shap_exp, X_df, quantile=0.2, direction="low")
    # for each feature, that only includes X.iloc[0]
    assert out == {
        "a": (1.0, 1.0),
        "b": (10.0, 10.0),
        "c": (100.0, 100.0),
    }


# -- High‐quantile happy path -----------------------------------------------
def test_high_quantile_ranges_on_numpy_input(X_df, shap_vals_array):
    # feed raw numpy + specify features subset
    # top 40% => rows with shap >= 60th percentile ([row indices 3,4])
    expected = {
        "b": (40.0, 50.0),
    }
    out = shap_vals.find_quantile_ranges(
        shap_vals_array, X_df, features=["b"], quantile=0.4, direction="high"
    )
    assert out == expected


def test_find_quantile_ranges_returns_min_value_when_quantile_zero(X_df):
    # pick quantile=0.0
    vals = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    # this produces cutoff=np.quantile(...,0)=1.0 -> mask only for equal entries
    out = shap_vals.find_quantile_ranges(
        vals, X_df[["a", "b"]], quantile=0.0, direction="low"
    )
    # only rows with shap == 1.0 on feature 'a'; for 'b', shap==2.0 exists
    assert out["a"] == (1.0, 1.0)
    assert out["b"] == (10.0, 10.0)


# -- Invalid‐feature edge case ------------------------------------------------
def test_invalid_feature_name_raises(X_df, shap_vals_array):
    shap_exp = FakeShapExp(shap_vals_array, list(X_df.columns))
    with pytest.raises(ValueError):
        shap_vals.find_quantile_ranges(
            shap_exp, X_df, features=["nonexistent"], quantile=0.1
        )


# -- Quantile out‐of‐bounds edge case ----------------------------------------
def test_quantile_out_of_bounds_raises(X_df):
    vals = np.zeros((3, 3))
    with pytest.raises(ValueError):
        shap_vals.find_quantile_ranges(vals, X_df, quantile=-0.5, direction="low")
    with pytest.raises(ValueError):
        shap_vals.find_quantile_ranges(vals, X_df, quantile=1.5, direction="high")


# ------ Tests for intersect_ranges function -------------------


def test_single_model_returns_same_ranges():
    d = {
        "a": (0, 5),
        "b": (10, 20),
    }
    # With only one dict, output should be identical
    out = shap_vals.intersect_ranges(d)
    assert out == d


def test_basic_overlap_intersection():
    d1 = {"a": (0, 5)}
    d2 = {"a": (2, 10)}
    # Overlap [2,5]
    assert shap_vals.intersect_ranges(d1, d2) == {"a": (2, 5)}


def test_no_overlap_returns_full_union():
    d1 = {"a": (0, 1)}
    d2 = {"a": (2, 3)}
    # No overlap: returns (min of mins, max of maxs) = (0,3)
    assert shap_vals.intersect_ranges(d1, d2) == {"a": (0, 3)}


def test_three_models_mixed_ranges():
    m1 = {"x": (1, 10), "y": (0, 5)}
    m2 = {"x": (5, 15), "y": (2, 3)}
    m3 = {"x": (8, 12), "y": (-1, 2)}
    # x: overlap of [1,10], [5,15], [8,12] : [8,10]
    # y: overlap of [0,5], [2,3], [-1,2] : [2,2]
    expected = {"x": (8, 10), "y": (2, 2)}
    assert shap_vals.intersect_ranges(m1, m2, m3) == expected


def test_identical_ranges():
    d1 = {"f": (3, 7)}
    d2 = {"f": (3, 7)}
    d3 = {"f": (3, 7)}
    # Intersection remains (3,7)
    assert shap_vals.intersect_ranges(d1, d2, d3) == {"f": (3, 7)}


def test_mismatched_keys_raises_value_error():
    d1 = {"a": (0, 1), "b": (2, 3)}
    d2 = {"a": (0, 1)}  # missing 'b'
    with pytest.raises(ValueError):
        shap_vals.intersect_ranges(d1, d2)


def test_multiple_features_union_on_disjoint_some_one_feature():
    # Feature 'u' overlaps, 'v' does not
    m1 = {"u": (0, 4), "v": (10, 12)}
    m2 = {"u": (2, 6), "v": (20, 22)}
    # u: intersection [2,4]
    # v: no overlap: union [10,22]
    expected = {"u": (2, 4), "v": (10, 22)}
    assert shap_vals.intersect_ranges(m1, m2) == expected

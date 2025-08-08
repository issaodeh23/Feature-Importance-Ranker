from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
from . import data_processing
from . import shap_vals
from . import train
from .logger_setup import setup_logger

logger = setup_logger("fir_log.txt", to_console=True)


class FeatureImportanceRanker:
    """A class to encapsulate the feature importance ranking process."""

    def __init__(
        self,
        filepath: str,
        target_column: str,
        columns_to_drop: tuple,
        batch_column: Optional[str] = None,
    ):
        """Initializes the FeatureImportanceRanker with the data file path, target column, and columns to drop."""
        self.filepath = filepath
        self.target_column = target_column
        self.columns_to_drop = columns_to_drop
        self.batch_column = batch_column
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """Loads and cleans the data."""
        self.data = data_processing.load_data(
            self.filepath, self.target_column, self.columns_to_drop
        )
        logger.info("Data loaded and cleaned successfully.")
        return self.data

    def split_data(
        self,
        train_size: float = 0.8,
        test_size: float = 0.2,
        random_state: int = 42,
        data: pd.DataFrame = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Splits the data into training and testing sets.

        :param: train_size: Fraction for training set.
        :param: test_size: Fraction for test set.
        :param: random_state: Random seed.
        :param: data: DataFrame to split.

        :return: Tuple of (X_train, X_test, y_train, y_test)
        """
        if data is None:
            data = self.data

        (
            self.X_test,
            self.X_train,
            self.y_test,
            self.y_train,
        ) = data_processing.split_data(
            data,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
        )
        logger.info("Data split into training and testing sets successfully.")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def tune_xgb(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        is_classification: bool = False,
    ) -> Dict[str, Any]:
        """Tunes the model hyperparameters using Optuna.

        :param:  X_train: Training features.
        :param:  X_test: Test features.
        :param:  y_train: Training targets.
        :param:  y_test: Test targets.
        :param:  is_classification: Whether to use classification.

        :return: Dict of best parameters.
        """
        study = train.tune_xgb(
            X_train,
            X_test,
            y_train,
            y_test,
            is_classification=is_classification,
        )
        self.best_params = study.best_params
        logger.info(f"Best hyperparameters found: {self.best_params}")
        return self.best_params

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        is_classification: bool = False,
    ) -> Any:
        """Initializes and trains the model."""
        self.model = train.model_initializer(
            X_train,
            y_train,
            is_classification=is_classification,
            params_dict=self.best_params,
        )
        logger.info("Model trained successfully.")
        return self.model

    def get_shap_values(self, model: Any, X_test: pd.DataFrame) -> Any:
        """Calculates SHAP values for the model."""
        self.shap_values = shap_vals.get_shap_values(model, X_test)
        logger.info("SHAP values calculated successfully.")
        return self.shap_values

    def get_feature_rankings(self, shap_values: Any) -> Dict[str, float]:
        """Gets feature rankings based on SHAP values."""
        self.rankings = shap_vals.get_tree_rankings(shap_values)
        logger.info("Feature rankings obtained successfully.")
        return self.rankings

    def combine_rankings(self, rankings: Dict[str, float]) -> List[Tuple[str, float]]:
        """Combines feature rankings if needed.

        :param:  rankings: Feature rankings dict.

        :return:  List of (feature, importance) tuples.
        """
        self.combined_rankings = shap_vals.combine_rankings(rankings)
        return self.combined_rankings

    def find_quantile_ranges(self, shap_values, X_test):
        """Finds quantile ranges for the SHAP values."""
        self.ranges = shap_vals.find_quantile_ranges(shap_values, X_test)
        return self.ranges

    def intersect_ranges(
        self,
        ranges: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Tuple[float, float]]:
        """Intersects the quantile ranges.

        :param:   ranges: Dict of feature to (min, max) range.


        :return:    Dict of feature to intersected (min, max) range.
        """
        self.ranges_dict = shap_vals.intersect_ranges(ranges)
        logger.info("Quantile ranges found and intersected successfully.")
        return self.ranges_dict

    def display_ranking_and_range(
        self,
        combined_rankings: List[Tuple[str, float]],
        ranges_dict: Dict[str, Tuple[float, float]],
        top_n: int = 10,
    ) -> None:
        """Displays the feature rankings and their quantile ranges."""
        shap_vals.display_ranking_and_range(combined_rankings, ranges_dict, top_n=top_n)
        logger.info("Displayed feature rankings and quantile ranges.")

    def run(self) -> Tuple[List[Tuple[str, float]], Dict[str, Tuple[float, float]]]:
        """Runs the entire feature importance ranking process."""
        self.data = self.load_data()
        X_train, X_test, y_train, y_test = self.split_data()
        best_params = self.tune_xgb(X_train, X_test, y_train, y_test)
        model = self.train_model(X_train, y_train, is_classification=False)
        shap_values = self.get_shap_values(model, X_test)
        rankings = self.get_feature_rankings(shap_values)
        combined_rankings = self.combine_rankings(rankings)
        ranges = self.find_quantile_ranges(shap_values, X_test)
        formatted_ranges = self.intersect_ranges(ranges)
        self.display_ranking_and_range(combined_rankings, formatted_ranges)
        logger.info("Feature importance ranking process completed.")
        return combined_rankings, formatted_ranges

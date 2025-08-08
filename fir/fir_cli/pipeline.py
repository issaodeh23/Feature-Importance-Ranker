import click
import yaml
import optuna
from ..core import FeatureImportanceRanker
from ..logger_setup import setup_logger

logger = setup_logger("fir_log.txt", to_console=True)


def pipeline(config_path: str = "tool_configuration.yaml"):
    """Runs the entire FIR pipeline using parameters from a YAML config file."""

    # Load config from YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info("=== Feature Importance Ranking Pipeline ===")

    # Extract parameters from config
    filepath = config["filepath"]
    target_column = config["target_column"]
    columns_to_drop = tuple(config.get("columns_to_drop", ()))
    train_size = config.get("train_size", 0.8)
    test_size = config.get("test_size", 0.2)
    random_state = config.get("random_state", 42)
    is_classification = config.get("is_classification", False)

    # Initialize the FeatureImportanceRanker
    ranker = FeatureImportanceRanker(filepath, target_column, columns_to_drop)

    # Load and clean data
    logger.info("[1/7] Loading and cleaning data...")
    ranker.load_data()
    logger.info("✔ Data loaded and cleaned successfully.")

    # Split data
    logger.info("[2/7] Splitting data into training and testing sets...")
    ranker.split_data(
        train_size=train_size, test_size=test_size, random_state=random_state
    )
    logger.info("✔ Data split into training and testing sets successfully.")

    # Tune model hyperparameters
    logger.info("[3/7] Hyperparameter tuning for XGBoost model...")
    ranker.tune_xgb(
        ranker.X_train,
        ranker.X_test,
        ranker.y_train,
        ranker.y_test,
        is_classification=is_classification,
    )
    logger.info("✔ Best hyperparameters found.")

    # Initialize and train model
    logger.info("[4/7] Initializing and training model...")
    ranker.train_model(
        ranker.X_train, ranker.y_train, is_classification=is_classification
    )
    logger.info("✔ Model trained successfully.")

    # Get SHAP values
    logger.info("[5/7] Computing SHAP values...")
    shap_values = ranker.get_shap_values(ranker.model, ranker.X_test)
    logger.info("✔ SHAP values computed successfully.")

    # Get feature rankings
    logger.info("[6/7] Computing feature rankings...")
    rankings = ranker.get_feature_rankings(shap_values)
    combined_rankings = ranker.combine_rankings(rankings)
    logger.info("✔ Feature rankings computed successfully.")

    # Get and display ranges
    logger.info("[7/7] Computing optimal feature ranges...")
    ranges = ranker.find_quantile_ranges(shap_values, ranker.X_test)
    ranges_dict = ranker.intersect_ranges(ranges)
    logger.info("✔ Quantile ranges computed successfully.")

    # Display results (this will log the table)
    logger.info("=== Feature Importance Ranking ===")
    ranker.display_ranking_and_range(combined_rankings, ranges_dict, top_n=10)

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    # Optionally, allow passing a config path as a command-line argument
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "tool_configuration.yaml"
    pipeline(config_path)

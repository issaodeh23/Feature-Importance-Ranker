from fir.core import FeatureImportanceRanker

# Initialize the FeatureImportanceRanker with the dataset and parameters
# This automatically loads and cleans the data and sets ranker.data equal to the cleaned DataFrame
ranker = FeatureImportanceRanker(
    filepath="/Users/issaodeh/Documents/GitHub/intern-project-FE/dataset_uv_line1.csv",
    target_column="QM_database.scope_scrap_rate",
    columns_to_drop=[
        "__index_level_0__",
        "QM_database.cast_scrap_rate",
        "QM_database.sand_scrap_rate",
    ],
)

# Load and clean the data
# This returns the cleaned DataFrame and sets it as an attribute of the ranker object
ranker.load_and_clean_data()


# Split the data into training and testing sets
# Returns X_train, X_test, y_train, y_test and also sets them as attributes of the ranker object
ranker.split_data(train_size=0.002, test_size=0.002, random_state=42)

# Tune the model hyperparameters using Optuna
# Returns the best hyperparameters and sets them as an attribute of the ranker object
# Tunes an XGBRegressor by default, but can be set to XGBClassifier by passing is_classification=True
ranker.tune_xgb(
    ranker.X_train,
    ranker.X_test,
    ranker.y_train,
    ranker.y_test,
    is_classification=False,
)

# Initialize and train the model with the best hyperparameters and sets it as an attribute of the ranker object
# Model is XGBRegressor by default, but can be set to XGBClassifier by passing is_classification=True
ranker.train_model(ranker.X_train, ranker.y_train, is_classification=False)

# Get SHAP values for the model on the test set
# This returns the SHAP values and sets them as an attribute of the ranker object
shap_values = ranker.get_shap_values(ranker.model, ranker.X_test)

# Get feature rankings based on SHAP values
# This returns the rankings and sets them as an attribute of the ranker object
rankings = ranker.get_feature_rankings(shap_values)

# Combine rankings if needed (currently just returns the same rankings)
# This returns the combined rankings and sets them as an attribute of the ranker object
combined_rankings = ranker.combine_rankings(rankings)

# Find quantile ranges for the SHAP values
# This returns the ranges and sets them as an attribute of the ranker object
ranges = ranker.find_quantile_ranges(shap_values, ranker.X_test)

# Intersect the quantile ranges to find common ranges across features
# This returns a dictionary of ranges and sets it as an attribute of the ranker object
ranges_dict = ranker.intersect_ranges(ranges)

# Display the top 10 feature rankings and their ranges
ranker.display_ranking_and_range(combined_rankings, ranges_dict, top_n=10)

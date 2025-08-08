

# Feature Importance Ranking (FIR) Tool

A Python package and CLI for **data cleaning**, **model training**, and **interpretable feature importance analysis** using SHAP values. FIR helps you identify which features most influence your target variable and what value ranges are optimal for those features given a tabular dataset.

---

## Features

- **Data Loading & Cleaning:**
  Import CSV or Parquet data, drop unwanted or constant columns, and handle missing values.
- **Model Training:**
  Train XGBoost regression or classification models with automated hyperparameter tuning (Optuna).
- **SHAP Value Computation:**
  Compute SHAP values for model interpretability.
- **Feature Importance Ranking:**
  Aggregate SHAP values to rank features by importance.
- **Optimal Feature Ranges:**
  Identify value ranges for top features associated with optimal outcomes.
- **CLI & Python API:**
  Use via command line or as a Python library.

---

## Installation

1. **Install FIR in editable mode (from intern-project-fe directory)**
   ```bash
   pip install -e .
   ```

---

## Usage


#### TOOL CONFIGURATION
```bash
fill the tool_configuration.yaml file in the home directory with the appropriate information
```


### INSTALL CLI
```bash
install CLI in editable mode by running the following command:
pip install -e .
```
### RUN CLI
```bash
python -m fir.fir_cli.pipeline
```
---

### Python API

```python
from fir.core import FeatureImportanceRanker

# Initialize the ranker
ranker = FeatureImportanceRanker(
    filepath="data/dataset.csv",
    target_column="target_col",
    columns_to_drop=("col1", "col2"),
)

# Load and clean data
ranker.load_data()

# Split data
ranker.split_data(train_size=0.8, test_size=0.2, random_state=42)

# Hyperparameter tuning
ranker.tune_xgb(
    ranker.X_train, ranker.X_test, ranker.y_train, ranker.y_test, is_classification=False
)

# Train model
ranker.train_model(ranker.X_train, ranker.y_train, is_classification=False)

# Compute SHAP values
shap_values = ranker.get_shap_values(ranker.model, ranker.X_test)

# Feature ranking
rankings = ranker.get_feature_rankings(shap_values)
combined_rankings = ranker.combine_rankings(rankings)

# Optimal value ranges
ranges = ranker.find_quantile_ranges(shap_values, ranker.X_test)
ranges_dict = ranker.intersect_ranges(ranges)

# Display results
ranker.display_ranking_and_range(combined_rankings, ranges_dict, top_n=10)
```



---

## Customization

- Change the number of top features or quantile thresholds for optimal range extraction.
- Adjust train/test split sizes as needed.
- Adapts for both classification and regression datasets
- Extend with your own models or data cleaning steps.

---

## Output

- **Feature Importance Ranking:**
  Features sorted by their impact on the target variable.
- **Optimal Ranges:**
  Value ranges for each top feature associated with best model outcomes.
- **Results Saved:**
Results are saved in a txt file named fir_log.txt where all logs displayed in the terminal will be saved for ease of future manipulation.
---

## Notes

- **Notebooks**
  All notebooks are intended to try out specific variations of the tool in a jupyter notebook environment.

---
## License

This project is for internal use and research. Contact the author for external use or distribution.

---

## Authors

Made with ❤️ by [Nameer Jabara](https://github.com/Nameer-Jabara) & [Issa Odeh](https://github.com/issaodeh23)

---

## Acknowledgements

- [SHAP](https://github.com/slundberg/shap)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)

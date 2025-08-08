import yaml
from fir.core import FeatureImportanceRanker

with open("tool_configuration.yaml", "r") as f:
    config = yaml.safe_load(f)

ranker = FeatureImportanceRanker(
    filepath=config["filepath"],
    target_column=config["target_column"],
    columns_to_drop=tuple(config.get("columns_to_drop", ())),
)

# Optionally, you can pass train_size, test_size, random_state, etc. to split_data
ranker.run()

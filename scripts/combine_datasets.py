from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILES = (
	BASE_DIR / "nsrdb_features_cache.csv",
	BASE_DIR / "sddb_features_cache.csv",
)
OUTPUT_FILE = BASE_DIR / "hrv_features_dataset.csv"


def combine_datasets() -> pd.DataFrame:
	"""Combine the NSRDB and SDDB feature rows into one training dataset."""
	missing_files = [path.name for path in INPUT_FILES if not path.exists()]
	if missing_files:
		raise FileNotFoundError(
			"Missing input file(s): " + ", ".join(missing_files)
		)

	datasets = [pd.read_csv(path) for path in INPUT_FILES]

	first_columns = list(datasets[0].columns)
	for path, dataset in zip(INPUT_FILES[1:], datasets[1:]):
		if list(dataset.columns) != first_columns:
			raise ValueError(
				f"Column mismatch between {INPUT_FILES[0].name} and {path.name}"
			)

	combined = pd.concat(datasets, ignore_index=True)
	combined = combined.drop_duplicates().reset_index(drop=True)
	combined.to_csv(OUTPUT_FILE, index=False)
	return combined


if __name__ == "__main__":
	combined_dataset = combine_datasets()
	print(f"Combined dataset saved to: {OUTPUT_FILE.name}")
	print(f"Rows: {len(combined_dataset)}")
	print(f"Columns: {len(combined_dataset.columns)}")
	if "label" in combined_dataset.columns:
		print("Label counts:")
		print(combined_dataset["label"].value_counts().sort_index().to_string())

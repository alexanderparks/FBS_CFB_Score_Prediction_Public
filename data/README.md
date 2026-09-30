# Data Directory

This directory is populated by the notebooks.

- `raw/`: Original JSON responses from CollegeFootballData. These files are cached locally and ignored by Git.
- `interim/`: Normalized game, team-game, advanced statistic, talent, recruiting, and betting-line tables.
- `processed/`: The final leakage-safe modeling table.

The historical data can be recreated by running `01_data_acquisition.ipynb` through `03_feature_engineering.ipynb`. The API key is loaded from `.env` and is never stored in a notebook or dataset.

Raw, interim, and row-level modeling data are local generated artifacts and are ignored by Git rather than redistributed. Users must obtain their own CollegeFootballData API key and comply with the provider's current terms.

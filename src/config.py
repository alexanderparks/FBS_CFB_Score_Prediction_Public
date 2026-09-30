from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"
PREDICTION_DIR = OUTPUT_DIR / "weekly_predictions"
MODEL_DIR = PROJECT_ROOT / "models"

START_SEASON = 2014
CURRENT_SEASON = 2026
FINAL_TEST_SEASON = 2025
RANDOM_STATE = 42


def create_project_directories():
    """Create the folders used by the notebooks if they do not already exist."""
    directories = [
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        FIGURE_DIR,
        EVALUATION_DIR,
        PREDICTION_DIR,
        MODEL_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


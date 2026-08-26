from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
REPORT_DIR = BASE_DIR / "reports"
PLOT_DIR = BASE_DIR / "generated_plots"
MODEL_DIR = BASE_DIR / "saved_models"

for folder in (DATASET_DIR, REPORT_DIR, PLOT_DIR, MODEL_DIR):
    folder.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


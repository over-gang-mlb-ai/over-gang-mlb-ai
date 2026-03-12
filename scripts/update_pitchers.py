import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.data_manager import DataManager


if __name__ == "__main__":
    df = DataManager.update_pitcher_stats()
    print(f"Done. Rows: {0 if df is None else len(df)}")

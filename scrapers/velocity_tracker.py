import pandas as pd
from functools import lru_cache
from pathlib import Path
from unidecode import unidecode

_VELOCITY_CSV = Path(__file__).resolve().parent.parent / "data" / "velocity_data.csv"


class VelocityTracker:
    _velocity_df = None

    @staticmethod
    def load_velocity_csv():
        if VelocityTracker._velocity_df is None:
            try:
                df = pd.read_csv(_VELOCITY_CSV)
                df["Name"] = df["Name"].apply(lambda x: unidecode(x.lower().strip()))
                VelocityTracker._velocity_df = df.set_index("Name")
                print(f"✅ Loaded velocity data for {len(df)} pitchers")
            except Exception as e:
                print(f"⚠️ Could not load velocity_data.csv: {e}")
                VelocityTracker._velocity_df = pd.DataFrame()
        return VelocityTracker._velocity_df

    @staticmethod
    @lru_cache(maxsize=300)
    def get_velocity_drop(pitcher_name: str) -> float:
        try:
            norm_name = unidecode(pitcher_name.lower().strip())
            df = VelocityTracker.load_velocity_csv()
            if norm_name in df.index:
                season = float(df.loc[norm_name, "Season_Velo"])
                recent = float(df.loc[norm_name, "Recent_Velo"])
                drop = round(season - recent, 2)
                print(f"✅ Velo drop for {pitcher_name}: {drop}")
                return drop
            else:
                print(f"⚠️ Velo not found for {pitcher_name} in velocity_data.csv — using 0")
                return 0.0
        except Exception as e:
            print(f"⚠️ VelocityTracker error for {pitcher_name}: {e}")
            return 0.0

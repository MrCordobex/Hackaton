from typing import List
import pandas as pd

def load_csvs(paths: List[str]) -> List[pd.DataFrame]:
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            dfs.append(df)
        except Exception as e:
            # No levantamos excepción para no romper la app
            print(f"Error leyendo {p}: {e}")
    return dfs

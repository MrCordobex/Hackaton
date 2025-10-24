import os
from src.data_loader import load_csvs

def test_load_csvs_handles_missing(tmp_path):
    # Debe no romperse si el archivo no existe
    dfs = load_csvs([os.path.join(str(tmp_path), "no_existe.csv")])
    assert isinstance(dfs, list)

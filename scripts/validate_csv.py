#!/usr/bin/env python3
import sys, pandas as pd

def main(path: str):
    df = pd.read_csv(path)
    print(f"Archivo: {path}")
    print(f"Filas: {len(df)} | Columnas: {len(df.columns)}")
    print("Columnas:", ", ".join(df.columns))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python scripts/validate_csv.py data/raw/archivo.csv")
        sys.exit(1)
    main(sys.argv[1])

import os
import pandas as pd
import streamlit as st
from src.data_loader import load_csvs
from src.utils import human_bytes

st.set_page_config(page_title="Panel Mantenimiento", page_icon="🛠️", layout="wide")

st.title("🛠️ Panel de Mantenimiento")
st.caption("Interfaz para técnicos. Datos desde CSV en `data/raw/`.")

data_dir = os.getenv("DATA_DIR", "data/raw")
files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")] if os.path.isdir(data_dir) else []

with st.sidebar:
    st.header("📁 Datos")
    st.write(f"Ruta: `{data_dir}`")
    if not files:
        st.warning("No se encontraron CSV en la carpeta. Carga algunos para empezar.")
    else:
        st.success(f"{len(files)} archivos encontrados.")
        for f in files:
            full = os.path.join(data_dir, f)
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            st.write(f"- {f} ({human_bytes(size)})")

uploaded = st.file_uploader("Cargar CSV adicional (opcional)", type=["csv"], accept_multiple_files=True)
dfs = []

if files:
    dfs.extend(load_csvs([os.path.join(data_dir, f) for f in files]))

if uploaded:
    dfs.extend([pd.read_csv(f) for f in uploaded])

if not dfs:
    st.info("Sube o coloca CSV en `data/raw/` para ver datos.")
    st.stop()

# Combinar por columnas coincidentes (outer join por índice si no coincide)
try:
    # Heurística simple: concatenar por filas si columnas iguales; si no, st.tabs por archivo
    same_schema = all(set(dfs[0].columns) == set(df.columns) for df in dfs)
    if same_schema:
        df = pd.concat(dfs, ignore_index=True)
        st.success(f"Datos combinados: {df.shape[0]} filas × {df.shape[1]} columnas")
        st.dataframe(df, use_container_width=True)
    else:
        tabs = st.tabs([f"Archivo {i+1}" for i in range(len(dfs))])
        for t, df in zip(tabs, dfs):
            with t:
                st.write(f"{df.shape[0]} filas × {df.shape[1]} columnas")
                st.dataframe(df, use_container_width=True)
except Exception as e:
    st.error(f"Error combinando datos: {e}")

import streamlit as st
import pandas as pd
import os
from src.data_loader import load_csvs

st.title("📊 Análisis rápido")

data_dir = os.getenv("DATA_DIR", "data/raw")
files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".csv")] if os.path.isdir(data_dir) else []

if not files:
    st.info("No hay CSV en data/raw. Añade algunos para continuar.")
    st.stop()

dfs = load_csvs(files)

# Heurística: usar el primer CSV para KPIs
df = dfs[0]

st.subheader("KPIs básicos (auto)")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
if numeric_cols:
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Filas", len(df))
    with k2:
        st.metric("Columnas numéricas", len(numeric_cols))
    with k3:
        st.metric("Nulos totales", int(df.isna().sum().sum()))
else:
    st.info("No se detectaron columnas numéricas en el primer CSV.")

st.subheader("Vista previa")
st.dataframe(df.head(50), use_container_width=True)

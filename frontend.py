# frontend.py
import os
import json
import time
import streamlit as st

st.set_page_config(page_title="Incidencias", layout="centered")
st.title("Incidencias")

INCIDENTS_FILE = os.path.join("data", "incidents.json")

def load_incidents():
    if not os.path.exists(INCIDENTS_FILE):
        return []
    try:
        with open(INCIDENTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                return []
            # Orden cronológico: más recientes primero
            data.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return data
    except Exception:
        return []

def save_incidents(incidents):
    os.makedirs(os.path.dirname(INCIDENTS_FILE), exist_ok=True)
    tmp = INCIDENTS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(incidents, f, ensure_ascii=False, indent=2)
    os.replace(tmp, INCIDENTS_FILE)

# Lista
incidents = load_incidents()

if not incidents:
    st.info("No hay incidencias todavía.")
else:
    for idx, inc in enumerate(incidents):
        ts = inc.get("created_at", "—")
        comment = inc.get("comment", "—")
        col_text, col_btn = st.columns([10, 1])
        with col_text:
            st.markdown(f"- **{ts}** — {comment}")
        with col_btn:
            if st.button("Borrar", key=f"del_{idx}", help="Borrar esta incidencia"):
                new_list = [
                    i for i in incidents
                    if not (i.get("created_at") == ts and i.get("comment") == comment)
                ]
                save_incidents(new_list)
                st.rerun()

# Auto-refresco fijo (2 s)
time.sleep(2.0)
st.rerun()

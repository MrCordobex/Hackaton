# incident_generator.py
import os
import json
from datetime import datetime
import streamlit as st

DATA_DIR = "data"
INCIDENTS_FILE = os.path.join(DATA_DIR, "incidents.json")

st.set_page_config(page_title="Generar incidencia", layout="centered")
st.title("✍️ Nueva incidencia")

# --- almacenamiento mínimo ---
os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(INCIDENTS_FILE):
    with open(INCIDENTS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False)

def load_incidents():
    try:
        with open(INCIDENTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def save_incidents(incidents):
    tmp = INCIDENTS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(incidents, f, ensure_ascii=False, indent=2)
    os.replace(tmp, INCIDENTS_FILE)

# --- callback: guarda y limpia textarea ---
def on_save():
    text = (st.session_state.get("comment_text") or "").strip()
    if not text:
        st.session_state["flash_msg"] = ("warn", "El comentario está vacío.")
        return

    incidents = load_incidents()
    incidents.append({
        "comment": text,
        "created_at": datetime.utcnow().isoformat() + "Z",
    })
    save_incidents(incidents)

    # limpiar textarea y mostrar feedback
    st.session_state["comment_text"] = ""
    st.session_state["flash_msg"] = ("ok", "Incidencia guardada en data/incidents.json.")

# --- UI ---
st.text_area(
    "Comentario",
    key="comment_text",
    height=200,
    placeholder="Escribe aquí el comentario del conductor…"
)

st.button("Guardar", type="primary", on_click=on_save)

# feedback no intrusivo
msg = st.session_state.get("flash_msg")
if msg:
    kind, text = msg
    if kind == "ok":
        st.success(text)
        st.toast(text)
    else:
        st.warning(text)
    # opcional: limpiar el flash para no repetir
    st.session_state.pop("flash_msg", None)

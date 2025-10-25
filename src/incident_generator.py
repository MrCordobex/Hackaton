# incident_generator.py
import os
import json
from datetime import datetime
import uuid
import streamlit as st

# --- 🔧 FIX IMPORTANTE (doble mapeo para pickle/joblib) ---
import sys
import funciones as _funciones
sys.modules['__main__'] = _funciones
sys.modules['main'] = _funciones
# ----------------------------------------------------------

from aplicar_modelo_wrapper import aplicar_modelo_topk

# ===============================
# 📁 Archivos y configuración
# ===============================
DATA_DIR = "data"
INCIDENTS_FILE = os.path.join(DATA_DIR, "incidents.json")
PRED_FILE = os.path.join(DATA_DIR, "incidents_predichas.json")

st.set_page_config(page_title="Generar incidencia", layout="centered")
st.title("✍️ Nueva incidencia")

# ===============================
# 🧱 Inicialización de almacenamiento
# ===============================
os.makedirs(DATA_DIR, exist_ok=True)
for f in (INCIDENTS_FILE, PRED_FILE):
    if not os.path.exists(f):
        with open(f, "w", encoding="utf-8") as h:
            json.dump([], h, ensure_ascii=False)

def _load_list(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def _atomic_save(path, payload_list):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload_list, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_incidents():
    return _load_list(INCIDENTS_FILE)

def save_incidents(lst):
    _atomic_save(INCIDENTS_FILE, lst)

def load_predictions():
    return _load_list(PRED_FILE)

def save_predictions(lst):
    _atomic_save(PRED_FILE, lst)

# ===============================
# 🔁 Normalización de bundle Top-K (acepta varios formatos)
# ===============================
def _normalize_bundle(bundle: dict) -> dict:
    """
    Devuelve SIEMPRE:
    {
      "clavero": <str>,
      "accion": <str>,
      "topk": {"claveros": [ {"label": str, "score": float|None, "acciones": [{"label": str, "score": float|None}, ...]}, ... ]}
    }
    """
    if not isinstance(bundle, dict):
        raise RuntimeError("El wrapper no devolvió un dict.")

    # Formato A (tu wrapper actual): {"clavero","accion","topk":{"claveros":[...]}}
    if "clavero" in bundle and "topk" in bundle:
        clavero = str(bundle.get("clavero", "") or "")
        accion  = str(bundle.get("accion", "") or "")
        topk    = (bundle.get("topk") or {})
        clavs   = (topk.get("claveros") or [])
        return {"clavero": clavero, "accion": accion, "topk": {"claveros": clavs}}

    # Formato B (wrapper alternativo): {"primary":{"clavero","accion"},"claveros":[...]}
    if "primary" in bundle and "claveros" in bundle:
        primary = bundle.get("primary") or {}
        clavs   = bundle.get("claveros") or []
        clavero = str(primary.get("clavero", "") or "")
        accion  = str(primary.get("accion", "") or "")
        return {"clavero": clavero, "accion": accion, "topk": {"claveros": clavs}}

    # Formato C (mínimo): solo top-k claveros y no top-1 explícito
    if "claveros" in bundle:
        clavs = bundle.get("claveros") or []
        if clavs:
            c0 = clavs[0]
            clavero = str(c0.get("label", "") or "")
            acc0 = (c0.get("acciones") or [])
            accion = str(acc0[0].get("label", "") if acc0 else "")
        else:
            clavero, accion = "", ""
        return {"clavero": clavero, "accion": accion, "topk": {"claveros": clavs}}

    # Si nada coincide, fallo controlado
    raise RuntimeError("Estructura Top-K desconocida (no trae 'topk' ni 'primary').")

# ===============================
# 💾 Lógica principal: guardar + predecir (Top-K)
# ===============================
def on_save():
    text = (st.session_state.get("comment_text") or "").strip()
    if not text:
        st.session_state["flash_msg"] = ("warn", "El comentario está vacío.")
        return

    # timestamp + id estable
    ts = datetime.utcnow().isoformat() + "Z"
    inc_id = str(uuid.uuid4())

    # 1️⃣ Guarda la incidencia "raw"
    incidents = load_incidents()
    incidents.append({"id": inc_id, "comment": text, "created_at": ts})
    save_incidents(incidents)

    # 2️⃣ Predice (Top-K) y guarda en incidents_predichas.json (formato único para el frontend)
    try:
        raw_bundle = aplicar_modelo_topk(text, k_clavero=3, k_accion=3)
        norm = _normalize_bundle(raw_bundle)

        preds = load_predictions()
        preds.append({
            "id": inc_id,
            "created_at": ts,
            "comment": text,
            "prediction": norm
        })
        save_predictions(preds)
        st.session_state["flash_msg"] = ("ok", "Incidencia guardada y predicciones (Top-K) generadas ✅")

    except Exception as e:
        # Si el modelo falla, al menos se guardó la incidencia original
        st.session_state["flash_msg"] = ("warn", f"Incidencia guardada, pero falló la predicción: {e}")

    # limpiar textarea
    st.session_state["comment_text"] = ""

# ===============================
# 🧠 Interfaz de usuario
# ===============================
st.text_area(
    "Comentario",
    key="comment_text",
    height=200,
    placeholder="Escribe aquí el comentario del operario…"
)

st.button("Guardar", type="primary", on_click=on_save)

# ===============================
# 🪶 Feedback no intrusivo
# ===============================
msg = st.session_state.get("flash_msg")
if msg:
    kind, text = msg
    if kind == "ok":
        st.success(text)
        st.toast(text)
    else:
        st.warning(text)
    st.session_state.pop("flash_msg", None)

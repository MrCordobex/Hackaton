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

# ✅ 5 preguntas (con texto a añadir si están en "Sí")
QUESTIONS = [
    {
        "id": "q1",
        "label": "¿Hay fuga o fallo en el antirretorno?",
        "auto_text": "Fuga/Fallo en el antirretorno",
    },
    {
        "id": "q2",
        "label": "¿Hay retardo o no se aplica el freno de estacionamiento?",
        "auto_text": "Retardo/No aplica freno de estacionamiento",
    },
    {
        "id": "q3",
        "label": "¿Está la sirga de estacionamiento rota?",
        "auto_text": "Sirga de estacionamiento rota",
    },
    {
        "id": "q4",
        "label": "¿Hay fallo en valvula antibloqueo?",
        "auto_text": "Fallo en válvula antibloqueo",
    },
    {
        "id": "q5",
        "label": "¿Hay fuga en valvula de freno de estacionamiento?",
        "auto_text": "Fuga en válvula de freno de estacionamiento",
    },
]
# "No" por defecto, obligatorio elegir (solo 2 opciones)
YES_NO_OPTIONS = ["No", "Sí"]
SEPARATOR = " · "  # texto que separa las auto-frases añadidas

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
# 🔁 Normalización de bundle Top-K
# ===============================
def _normalize_bundle(bundle: dict) -> dict:
    if not isinstance(bundle, dict):
        raise RuntimeError("El wrapper no devolvió un dict.")
    if "clavero" in bundle and "topk" in bundle:
        clavero = str(bundle.get("clavero", "") or "")
        accion  = str(bundle.get("accion", "") or "")
        topk    = (bundle.get("topk") or {})
        clavs   = (topk.get("claveros") or [])
        return {"clavero": clavero, "accion": accion, "topk": {"claveros": clavs}}
    if "primary" in bundle and "claveros" in bundle:
        primary = bundle.get("primary") or {}
        clavs   = bundle.get("claveros") or []
        clavero = str(primary.get("clavero", "") or "")
        accion  = str(primary.get("accion", "") or "")
        return {"clavero": clavero, "accion": accion, "topk": {"claveros": clavs}}
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
    raise RuntimeError("Estructura Top-K desconocida (no trae 'topk' ni 'primary').")

# ===============================
# 🔎 Helpers Sí/No
# ===============================
def _coerce_yesno_to_bool(value_str):
    if value_str == "Sí":
        return True
    if value_str == "No":
        return False
    return None

def _collect_answers():
    answers = {}
    for q in QUESTIONS:
        sel = st.session_state.get(f"yn_{q['id']}", YES_NO_OPTIONS[0])
        answers[q["id"]] = _coerce_yesno_to_bool(sel)
    return answers

def _all_answered(answers: dict) -> bool:
    return all(v is True or v is False for v in answers.values())

# ===============================
# ✍️ Auto-append al comentario
# ===============================
# Snapshot de selecciones para detectar cambios a "Sí"
if "yn_snapshot" not in st.session_state:
    st.session_state["yn_snapshot"] = {q["id"]: "No" for q in QUESTIONS}

def _append_phrase_once(phrase: str):
    """Añade phrase al final del comentario si no está ya presente."""
    current = st.session_state.get("comment_text", "").strip()
    if not current:
        st.session_state["comment_text"] = phrase
        return
    # evita duplicados simples (búsqueda literal)
    if phrase not in current:
        st.session_state["comment_text"] = current + SEPARATOR + phrase

def handle_yesno_change():
    """
    Se llama cada vez que cambia cualquier select.
    Si una pregunta pasa de != 'Sí' a 'Sí', añadimos su auto_text al comentario.
    Si pasa a 'No', no hacemos nada (solo añadimos, nunca borramos).
    """
    snapshot = st.session_state.get("yn_snapshot", {})
    for q in QUESTIONS:
        key = f"yn_{q['id']}"
        curr = st.session_state.get(key, "No")
        prev = snapshot.get(q["id"], "No")
        if curr == "Sí" and prev != "Sí":
            _append_phrase_once(q["auto_text"])
        # actualiza snapshot
        snapshot[q["id"]] = curr
    st.session_state["yn_snapshot"] = snapshot

# ===============================
# 💾 Guardar + predecir
# ===============================
def on_save():
    text = (st.session_state.get("comment_text") or "").strip()
    if not text:
        st.session_state["flash_msg"] = ("warn", "El comentario está vacío.")
        return

    yn_answers = _collect_answers()
    if not _all_answered(yn_answers):
        st.session_state["flash_msg"] = ("warn", "Responde Sí/No en todas las preguntas.")
        return

    ts = datetime.utcnow().isoformat() + "Z"
    inc_id = str(uuid.uuid4())

    incidents = load_incidents()
    incidents.append({
        "id": inc_id,
        "comment": text,
        "created_at": ts,
        "yes_no": yn_answers,
    })
    save_incidents(incidents)

    try:
        raw_bundle = aplicar_modelo_topk(text, k_clavero=3, k_accion=3)
        norm = _normalize_bundle(raw_bundle)
        preds = load_predictions()
        preds.append({
            "id": inc_id,
            "created_at": ts,
            "comment": text,
            "yes_no": yn_answers,
            "prediction": norm
        })
        save_predictions(preds)
        st.session_state["flash_msg"] = ("ok", "Incidencia guardada y predicciones (Top-K) generadas ✅")
    except Exception as e:
        st.session_state["flash_msg"] = ("warn", f"Incidencia guardada, pero falló la predicción: {e}")

    st.session_state["comment_text"] = ""

# ===============================
# 🧠 UI
# ===============================
st.subheader("Checklist rápido")
for q in QUESTIONS:
    st.selectbox(
        q["label"],
        options=YES_NO_OPTIONS,
        index=0,  # "No" por defecto
        key=f"yn_{q['id']}",
        help="Selecciona Sí o No.",
        on_change=handle_yesno_change,  # 👈 detecta cambios para auto-append
    )

st.text_area(
    "Comentario",
    key="comment_text",
    height=200,
    placeholder="Escribe aquí el comentario del operario…",
)

_current_answers = _collect_answers()
disable_save = (not st.session_state.get("comment_text")) or (not _all_answered(_current_answers))
st.button("Guardar", type="primary", on_click=on_save, disabled=disable_save)

# ===============================
# 🪶 Feedback
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

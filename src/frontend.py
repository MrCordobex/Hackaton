# frontend.py
import os
import json
import time
from datetime import datetime, date
import streamlit as st

# ============= FIX pickle (__main__/main) =============
import sys
import funciones as _funciones
sys.modules['__main__'] = _funciones
sys.modules['main'] = _funciones
# ======================================================

# Wrapper: Top-K inicial y recálculo de acciones con feedback
from aplicar_modelo_wrapper import (
    aplicar_modelo_topk,
    aplicar_modelo_recalc_acciones,
)

# ------------------- Config -------------------
st.set_page_config(page_title="Incidencias | Top-K", layout="wide")

st.title("📋 Incidencias")
st.caption("Clasificación con sugerencias Top-K y filtros por estado")

DATA_DIR = "data"
INCIDENTS_FILE = os.path.join(DATA_DIR, "incidents.json")
PRED_FILE = os.path.join(DATA_DIR, "incidents_predichas.json")
FEEDBACK_FILE = os.path.join(DATA_DIR, "incidents_feedback.json")

REFRESH_SECONDS = 2.0

# ---------------- Estado edición persistente ----------------
if "edit" not in st.session_state:
    st.session_state["edit"] = {}  # { "<tab>_<key>": bool }

def is_any_editing() -> bool:
    return any(bool(v) for v in st.session_state["edit"].values())

# ------------------- Estilos -------------------
st.markdown("""
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 1.2rem; }
.card {padding: 1rem 1.1rem; border: 1px solid #E5E7EB; border-radius: 14px; margin-bottom: 0.85rem; background: #fff;}
.card-header {display:flex; justify-content: space-between; align-items:center; margin-bottom:0.25rem;}
.meta {color:#6B7280; font-size:0.88rem;}
.pill {display:inline-block; padding: 0.2rem 0.55rem; border-radius: 9999px; background:#F3F4F6; border:1px solid #E5E7EB; font-size:0.85rem; margin-right:0.35rem;}
.badge-ok {background:#DCFCE7; border-color:#A7F3D0;}
.badge-pend {background:#E0E7FF; border-color:#C7D2FE;}
.badge-rev  {background:#FEF9C3; border-color:#FDE68A;}
.kv {font-size:0.96rem; margin: 0.25rem 0;}
.kv b {color:#111827}
.btn-row {display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap; margin-top:0.4rem;}
.sidebar-title {font-weight: 600; color:#111827;}
.topk-row {display:flex; flex-direction:column; gap:0.35rem;}
.topk-pill {display:inline-block; padding: 0.15rem 0.5rem; border-radius: 9999px; background:#F3F4F6; border:1px solid #E5E7EB; font-size:0.86rem; margin-right:0.35rem;}
.topk-small {color:#6B7280; font-size:0.85rem;}
</style>
""", unsafe_allow_html=True)

# ------------------- Utilidades JSON -------------------
def _load_list(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def _atomic_save(path, payload_list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload_list, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_predictions():
    data = _load_list(PRED_FILE)
    data.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return data

def save_predictions(lst):
    _atomic_save(PRED_FILE, lst)

def load_feedback():
    return _load_list(FEEDBACK_FILE)

def save_feedback(lst):
    _atomic_save(FEEDBACK_FILE, lst)

# --------------- Helpers robustos (ID/keys) ---------------
def make_safe_key(item):
    _id = item.get("id")
    ts = item.get("created_at", "")
    comment = item.get("comment", "")
    if _id:
        return _id
    return f"{ts}__{abs(hash(comment))}"

def same_item(a, b):
    if a.get("id") and b.get("id"):
        return a["id"] == b["id"]
    return (a.get("created_at") == b.get("created_at")) and (a.get("comment") == b.get("comment"))

def find_index_by_item(lst, target_item):
    for idx, it in enumerate(lst):
        if same_item(it, target_item):
            return idx
    return None

def fmt_score(score):
    if score is None:
        return "—"
    try:
        return f"{100.0*float(score):.1f}%"
    except Exception:
        return "—"

def _status_of(it) -> str:
    status = (it.get("status") or "").lower().strip()
    if status == "ok":
        return "Satisfechas"
    if (it.get("revisions", []) or []):
        return "Revisadas"
    return "Pendientes"

# ------------------- Filtros (sidebar) -------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🔎 Filtros</div>", unsafe_allow_html=True)
    preds_all = load_predictions()

    def _to_date(s):
        try:
            return datetime.fromisoformat(s.replace("Z","")).date()
        except Exception:
            return None

    min_date = None
    max_date = None
    for it in preds_all:
        d = _to_date(it.get("created_at",""))
        if d:
            min_date = d if (min_date is None or d < min_date) else min_date
            max_date = d if (max_date is None or d > max_date) else max_date
    if min_date is None: min_date = date.today()
    if max_date is None: max_date = date.today()

    start_d, end_d = st.date_input("Rango de fechas", value=(min_date, max_date))
    q_text = st.text_input("Texto en comentario", placeholder="palabras clave…").strip()

    # filtro por estado
    status_options = ["— todos —", "Pendientes", "Revisadas", "Satisfechas"]
    sel_status = st.selectbox("Estado", options=status_options, index=0)

    # filtros por clavero top-1
    claveros_top1 = sorted({ (it.get("prediction") or {}).get("clavero","") for it in preds_all if (it.get("prediction") or {}).get("clavero") })
    sel_clav = st.selectbox("Clavero (top-1)", options=["— todos —"] + claveros_top1, index=0)

    st.divider()
    st.caption("Auto-refresco cada ~2 s (pausa si estás editando)")

def _item_date_ok(it):
    try:
        d = datetime.fromisoformat(it.get("created_at","").replace("Z","")).date()
    except Exception:
        return False
    return (d >= start_d) and (d <= end_d)

def _item_text_ok(it):
    if not q_text: return True
    return q_text.lower() in (it.get("comment","") or "").lower()

def _item_clavero_ok(it):
    if sel_clav == "— todos —": return True
    return (it.get("prediction") or {}).get("clavero","") == sel_clav

def _item_status_ok(it):
    if sel_status == "— todos —": return True
    return _status_of(it) == sel_status

# ------------------- KPIs -------------------
total = len(preds_all)
num_ok = sum(1 for it in preds_all if _status_of(it) == "Satisfechas")
num_rev = sum(1 for it in preds_all if _status_of(it) == "Revisadas")
num_pending = sum(1 for it in preds_all if _status_of(it) == "Pendientes")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total", total)
c2.metric("Pendientes", num_pending)
c3.metric("Revisadas", num_rev)
c4.metric("Satisfechas", num_ok)
st.divider()

# ------------------- Render tarjeta -------------------
def _badge_html(it):
    s = _status_of(it)
    if s == "Satisfechas":
        return "<span class='pill badge-ok'>Satisfecha</span>"
    if s == "Revisadas":
        return "<span class='pill badge-rev'>Revisada</span>"
    return "<span class='pill badge-pend'>Pendiente</span>"

def render_card(it, key_prefix: str):
    safe_key = make_safe_key(it)
    ts = it.get("created_at","—")
    comment = it.get("comment","—")
    pred = it.get("prediction",{}) or {}
    top1_clav = pred.get("clavero","—")
    top1_acc  = pred.get("accion","—")
    topk = ((pred.get("topk") or {}).get("claveros") or [])

    edit_key = f"{key_prefix}_{safe_key}"

    st.markdown('<div class="card">', unsafe_allow_html=True)

    left, right = st.columns([6,2])
    with left:
        header_html = f"<div class='card-header'><div class='meta'>{ts}</div><div>{_badge_html(it)}</div></div>"
        st.markdown(header_html, unsafe_allow_html=True)
        st.markdown(f"<p class='kv'><b>Comentario</b>: {comment}</p>", unsafe_allow_html=True)
    with right:
        pass

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<p class='kv'><b>Clavero</b>: {top1_clav}</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<p class='kv'><b>Acción</b>: {top1_acc}</p>", unsafe_allow_html=True)

    # ------- Panel Top-K -------
    with st.expander("🔎 Ver Top-K sugerencias (3×3)", expanded=False):
        if not topk:
            calc_key = f"{key_prefix}_calc_topk_{safe_key}"
            if st.button("Calcular Top-K ahora", key=calc_key, help="Genera y guarda 3×3 para esta incidencia"):
                try:
                    new_pred = aplicar_modelo_topk(comment, k_clavero=3, k_accion=3)
                    if not isinstance(new_pred, dict) or "topk" not in new_pred:
                        raise RuntimeError("Wrapper no devolvió estructura Top-K válida.")
                    all_preds = load_predictions()
                    idx_match = find_index_by_item(all_preds, it)
                    if idx_match is not None:
                        all_preds[idx_match]["prediction"] = new_pred
                        save_predictions(all_preds)
                        st.toast("Top-K generado y guardado ✅")
                        st.rerun()
                    else:
                        st.warning("No se pudo localizar la incidencia para actualizar.")
                except Exception as e:
                    st.error(f"No se pudo calcular Top-K: {e}")
            st.caption("No hay Top-K disponible en esta incidencia.")
        else:
            # Lista de claveros con scores y sus acciones
            for ci, cblock in enumerate(topk):
                c_label = cblock.get("label","—")
                c_score = fmt_score(cblock.get("score"))
                st.markdown(f"**{ci+1}. {c_label}**  <span class='topk-small'>(conf: {c_score})</span>", unsafe_allow_html=True)

                acciones = cblock.get("acciones", []) or []
                if not acciones:
                    st.caption("• sin acciones sugeridas")
                    continue

                options = [f"{a.get('label','—')}  (conf: {fmt_score(a.get('score'))})" for a in acciones]
                rad_key = f"{key_prefix}_rad_{safe_key}_{ci}"
                choice = st.radio("Acciones", options=options, index=0, key=rad_key, horizontal=True, label_visibility="collapsed")

                apply_key = f"{key_prefix}_apply_{safe_key}_{ci}"
                if st.button("Aplicar esta opción", key=apply_key, help="Sustituir predicción por esta combinación"):
                    chosen_idx = options.index(choice)
                    new_clav = c_label
                    new_acc  = acciones[chosen_idx].get("label","")
                    all_preds = load_predictions()
                    idx_match = find_index_by_item(all_preds, it)
                    if idx_match is not None:
                        itm = all_preds[idx_match]
                        old_pred = itm.get("prediction",{}) or {}
                        old = {"clavero": old_pred.get("clavero","—"), "accion": old_pred.get("accion","—")}
                        revs = itm.get("revisions",[]) or []
                        revs.append({
                            "created_at": datetime.utcnow().isoformat() + "Z",
                            "old_prediction": old,
                            "correction_comment": "Selección manual desde Top-K",
                            "new_prediction": {"clavero": new_clav, "accion": new_acc}
                        })
                        itm["revisions"] = revs
                        itm["prediction"]["clavero"] = new_clav
                        itm["prediction"]["accion"] = new_acc
                        itm["status"] = "revised"
                        save_predictions(all_preds)
                        st.toast("Predicción actualizada desde Top-K ✅")
                        st.rerun()
                    else:
                        st.warning("No se pudo localizar la incidencia para actualizar.")

    # ------- Botones principales -------
    st.markdown("<div class='btn-row'>", unsafe_allow_html=True)
    ok_clicked  = st.button("✅ Satisfecho", key=f"{key_prefix}_ok_{safe_key}")
    ko_clicked  = st.button("✏️ Mejorar",  key=f"{key_prefix}_ko_{safe_key}")
    del_clicked = st.button("🗑️ Borrar pred.", key=f"{key_prefix}_del_{safe_key}")
    st.markdown("</div>", unsafe_allow_html=True)

    # OK → marcar estado y feedback simple
    if ok_clicked:
        fb = load_feedback()
        fb.append({
            "id": it.get("id"),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "original_comment": comment,
            "original_prediction": {"clavero": top1_clav, "accion": top1_acc},
            "user_feedback": "ok"
        })
        save_feedback(fb)

        all_preds = load_predictions()
        idx_match = find_index_by_item(all_preds, it)
        if idx_match is not None:
            all_preds[idx_match]["status"] = "ok"
            save_predictions(all_preds)
            st.toast("Marcada como satisfecha ✅")
        else:
            st.warning("No se pudo localizar la incidencia para actualizar.")
        st.rerun()

    # Borrado
    if del_clicked:
        all_preds = load_predictions()
        remaining = [x for x in all_preds if not same_item(x, it)]
        save_predictions(remaining)
        st.rerun()

    # Entrar en edición
    if ko_clicked:
        st.session_state["edit"][edit_key] = True

    # ------- Expander de corrección (recalcular SOLO actuaciones con feedback) -------
    if st.session_state["edit"].get(edit_key, False):
        with st.expander("✍️ Proponer corrección y recalcular actuaciones (Modelo 2)", expanded=True):
            draft_key = f"{key_prefix}_corr_{safe_key}"
            new_txt = st.text_area(
                "Describe la corrección o el contexto adicional",
                key=draft_key,
                placeholder="Ej: No es timonería; el problema real es el cableado del sensor XYZ…",
                height=120
            )
            c1, c2 = st.columns([1,1])
            with c1:
                recalc_clicked = st.button("🚀 Recalcular actuaciones (Top-K)", key=f"{key_prefix}_recalc_{safe_key}")
            with c2:
                cancel_clicked = st.button("Cancelar", key=f"{key_prefix}_cancel_{safe_key}")

            if cancel_clicked:
                st.session_state["edit"][edit_key] = False
                st.stop()

            if recalc_clicked:
                correction = (new_txt or "").strip()
                if not correction:
                    st.warning("Escribe una corrección para recalcular.")
                else:
                    fb = load_feedback()
                    fb_entry = {
                        "id": it.get("id"),
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "original_comment": comment,
                        "original_prediction": {"clavero": top1_clav, "accion": top1_acc},
                        "correction_comment": correction
                    }
                    try:
                        # Lista de claveros actual (mantener orden). Si no hay, usa el top-1.
                        claveros_exist = [c.get("label","") for c in topk] if topk else ([top1_clav] if top1_clav else [])

                        new_pred = aplicar_modelo_recalc_acciones(
                            descripcion_ot=comment,
                            comentario_feedback=correction,
                            claveros_existentes=claveros_exist,
                            top1_clavero=top1_clav,
                            k_accion=3
                        )
                        if not isinstance(new_pred, dict) or "clavero" not in new_pred:
                            raise RuntimeError("El wrapper no devolvió un diccionario válido en recalc_acciones.")

                        fb_entry["recalc"] = {"clavero": new_pred.get("clavero",""), "accion": new_pred.get("accion","")}

                        all_preds = load_predictions()
                        idx_match = find_index_by_item(all_preds, it)
                        if idx_match is not None:
                            itm = all_preds[idx_match]
                            revs = itm.get("revisions", []) or []
                            revs.append({
                                "created_at": datetime.utcnow().isoformat() + "Z",
                                "old_prediction": {"clavero": top1_clav, "accion": top1_acc},
                                "correction_comment": correction,
                                "new_prediction": {"clavero": new_pred.get("clavero",""), "accion": new_pred.get("accion","")}
                            })
                            itm["revisions"] = revs
                            itm["prediction"] = new_pred  # mismo conjunto de claveros, nuevas actuaciones
                            itm["status"] = "revised"
                            save_predictions(all_preds)
                            st.toast("Actuaciones recalculadas con feedback ✅")
                        else:
                            st.warning("No se pudo localizar la incidencia para revisar.")
                    except Exception as e:
                        fb_entry["error"] = str(e)
                        st.error(f"No se pudo recalcular: {e}")

                    fb.append(fb_entry)
                    save_feedback(fb)
                    st.session_state["edit"][edit_key] = True
                    st.rerun()

    # Histórico (sin f-strings con comillas conflictivas)
    revisions = it.get("revisions", []) or []
    if revisions:
        with st.expander("🕘 Histórico de revisiones"):
            for r in revisions:
                stamp  = r.get("created_at", "—")
                oldp   = r.get("old_prediction", {}) or {}
                newp   = r.get("new_prediction", {}) or {}
                reason = r.get("correction_comment", "—")

                st.markdown(f"**{stamp}**", unsafe_allow_html=True)
                st.markdown(f"- Antes → Clavero: `{oldp.get('clavero','—')}`, Acción: `{oldp.get('accion','—')}`")
                st.markdown(f"- Después → Clavero: `{newp.get('clavero','—')}`, Acción: `{newp.get('accion','—')}`")
                st.markdown(f"- Motivo: {reason}")

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------- Render principal -------------------
preds_all = load_predictions()
filtered = [
    it for it in preds_all
    if _item_date_ok(it) and _item_text_ok(it) and _item_clavero_ok(it) and _item_status_ok(it)
]

if not filtered:
    st.info("No hay incidencias con los filtros actuales.")
else:
    cols = st.columns(2)
    for idx, it in enumerate(filtered):
        with cols[idx % 2]:
            render_card(it, key_prefix="main")

# ------------------- Auto-refresh (pausado si editas) -------------------
if not is_any_editing():
    time.sleep(REFRESH_SECONDS)
    st.rerun()

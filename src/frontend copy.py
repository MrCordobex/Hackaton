# frontend.py
import os
import json
import time
from datetime import datetime, date
import streamlit as st

# --- dependencias opcionales / suaves ---
try:
    import pandas as pd
except Exception:
    pd = None

# ============= FIX pickle (__main__/main) =============
import sys
import funciones as _funciones
sys.modules['__main__'] = _funciones
sys.modules['main'] = _funciones
# ======================================================

# Wrappers
from aplicar_modelo_wrapper import (
    aplicar_modelo_topk,
    aplicar_modelo_recalc_acciones,
    aplicar_modelo_recalc_todo,  # ← NUEVO
)

# ===== Definiciones por "Código tarea std" (CLAVERO+ACCIÓN) =====
import csv, math, re, unicodedata
from pathlib import Path

DEF_CSV_PATH = str(Path(__file__).resolve().parent.parent / "data/processed/Definiciones_clave_Codigo_actuacion_V6_utf8.csv")
print("\n[DEBUG] Cargando definiciones desde:", DEF_CSV_PATH)

def _norm(s) -> str:
    """Normaliza texto y gestiona NaN o None → 'No hacer nada'."""
    if s is None:
        return "No hacer nada"
    if isinstance(s, float):
        if math.isnan(s):
            return "No hacer nada"
        s = str(s)
    s = str(s).strip()
    s = unicodedata.normalize("NFKC", s)
    return s

def _find_col(cols, *cands):
    low = {c.lower(): c for c in cols}
    for cand in cands:
        if cand.lower() in low:
            return low[cand.lower()]
    for c in cols:
        cl = c.lower()
        for cand in cands:
            if cand.lower() in cl:
                return c
    return None

@st.cache_data(show_spinner=False)
def load_defs_by_codigo_tarea_std(path: str):
    print("[DEBUG] Entrando en load_defs_by_codigo_tarea_std()")
    if not os.path.exists(path):
        print("[ERROR] CSV no encontrado:", path)
        return {}
    try:
        import pandas as pd
        print("[DEBUG] Intentando leer CSV con pandas...")
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        print(f"[DEBUG] CSV cargado. Forma: {df.shape}")
        print("[DEBUG] Columnas detectadas:", list(df.columns))

        col_code = _find_col(df.columns, "Código tarea std", "Codigo tarea std", "codigo tarea std", "codigo_tarea_std")
        col_def  = _find_col(df.columns,
                             "Definición", "Definicion", "DEFINICION",
                             "Definition", "DEFINITION",
                             "descripcion", "descripción")
        print(f"[DEBUG] Columna código detectada: {col_code}")
        print(f"[DEBUG] Columna definición detectada: {col_def}")
        if not col_code or not col_def:
            print("[ERROR] No se encontraron columnas adecuadas.")
            return {}

        df[col_code] = df[col_code].astype(str).map(_norm)
        df[col_def]  = df[col_def].astype(str).map(_norm)
        df = df.dropna(subset=[col_code]).drop_duplicates(subset=[col_code], keep="first")
        mapping = dict(zip(df[col_code], df[col_def]))
        print(f"[DEBUG] Diccionario construido. Claves cargadas: {len(mapping)}")
        # Mostrar algunas claves de ejemplo
        for i, (k, v) in enumerate(mapping.items()):
            if i >= 5:
                break
            print(f"  Ejemplo {i+1}: {k} -> {v[:60]}...")
        return mapping

    except Exception as e:
        print("[ERROR] Falló lectura con pandas:", e)

    print("[DEBUG] Intentando fallback con csv module...")
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            sample = fh.read(2048)
            fh.seek(0)
            dialect = csv.Sniffer().sniff(sample)
            reader = csv.DictReader(fh, dialect=dialect)
            cols = reader.fieldnames or []
            print("[DEBUG] Columnas detectadas (fallback):", cols)
            col_code = _find_col(cols, "Código tarea std", "Codigo tarea std", "codigo tarea std")
            col_def  = _find_col(cols, "Definición", "Definicion", "descripcion", "descripción", "DEFINICION")
            print(f"[DEBUG] Columna código: {col_code} | Columna definición: {col_def}")
            if not col_code or not col_def:
                print("[ERROR] No se encontraron columnas en fallback.")
                return {}
            mapping = {}
            for row in reader:
                code = _norm(row.get(col_code, ""))
                defi = _norm(row.get(col_def, ""))
                if code and defi and code not in mapping:
                    mapping[code] = defi
            print(f"[DEBUG] Fallback completado. Claves cargadas: {len(mapping)}")
            return mapping
    except Exception as e:
        print("[ERROR] Fallback CSV falló:", e)
        return {}

DEFS_BY_CODE = load_defs_by_codigo_tarea_std(DEF_CSV_PATH)
print(f"[DEBUG] Total de definiciones cargadas: {len(DEFS_BY_CODE)}")

def _variants(clavero: str, accion: str):
    c = _norm(clavero)
    a = _norm(accion)
    base = f"{c}{a}"
    yield base
    yield f"{c} {a}"
    yield f"{c}-{a}"
    yield f"{c}_{a}"
    yield re.sub(r"\W+", "", base)

def get_def_by_clavero_accion(clavero: str, accion: str) -> str:
    """Busca definición usando combinaciones típicas de CLAVERO+ACCIÓN."""
    if not clavero or not accion or accion.upper() == "NAN":
        return ""
    for key in _variants(clavero, accion):
        v = DEFS_BY_CODE.get(key)
        if v:
            print(f"[DEBUG] Match exacto: {key}")
            return v
    m = re.match(r"^([A-Za-z]+)(\d+)([A-Za-z0-9]*)$", _norm(clavero))
    if m:
        alt = f"{m.group(1)} {m.group(2)}{m.group(3)} {accion}"
        for key in _variants(alt, accion):
            v = DEFS_BY_CODE.get(key)
            if v:
                print(f"[DEBUG] Match alternativo: {key}")
                return v
    # búsqueda relajada
    c_norm = _norm(clavero)
    a_norm = _norm(accion)
    for code, defi in DEFS_BY_CODE.items():
        if code.startswith(c_norm) and code.endswith(a_norm):
            print(f"[DEBUG] Match parcial: {code}")
            return defi
    return ""
# ===============================================================

# ------------------- Config -------------------
st.set_page_config(page_title="Incidencias | Top-K", layout="wide")

# --- Session state ---
if "edit" not in st.session_state:
    st.session_state["edit"] = {}
if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = True

# ======= Branding CAF (rojo/blanco) =======
CAF_RED = "#E30613"
CAF_BG  = "#F8F9FB"

st.markdown(f"""
<style>
.main .block-container {{ padding-top: .6rem; }}
html, body, .main {{ background: {CAF_BG} !important; }}

.hero {{
  background: linear-gradient(90deg, {CAF_RED} 0%, #a90b13 100%);
  color: white; padding: 12px 18px; border-radius: 14px; margin-bottom: 12px;
  display:flex; align-items:center; gap:.6rem;
}}
.hero .logo {{ width: 28px; height: 28px; border-radius: 6px; background: white; color:{CAF_RED};
  display:flex; align-items:center; justify-content:center; font-weight:800; }}
.hero .title {{ font-size: 1.2rem; font-weight: 700; margin:0; }}
.hero .desc  {{ font-size: .92rem; opacity:.95; margin:0; }}

.card {{ padding: 1rem 1.1rem; border: 1px solid #E5E7EB; border-radius: 14px; 
  margin-bottom: .9rem; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
.card-header {{display:flex; justify-content: space-between; align-items:center; margin-bottom:.25rem;}}
.meta {{color:#6B7280; font-size:.88rem;}}
.pill {{display:inline-block; padding:.2rem .55rem; border-radius:9999px; background:#F3F4F6; border:1px solid #E5E7EB; font-size:.85rem; margin-right:.35rem;}}
.badge-ok  {{background:#DCFCE7; border-color:#A7F3D0;}}
.badge-pend{{background:#E0E7FF; border-color:#C7D2FE;}}
.badge-rev {{background:#FEF9C3; border-color:#FDE68A;}}
.kv {{font-size:.96rem; margin:.25rem 0;}}
.kv b {{color:#111827}}
.btn-row {{display:flex; gap:.5rem; align-items:center; flex-wrap:wrap; margin-top:.4rem;}}
.sidebar-title {{font-weight:700; color:{CAF_RED};}}
.topk-small {{color:#6B7280; font-size:.85rem;}}
.locked {{ border:1px dashed #CBD5E1; background:#F8FAFC; padding:.6rem .75rem; border-radius:12px; margin-top:.4rem; }}
.locked b {{ color:{CAF_RED}; }}

.stButton > button[kind="primary"] {{
  background: {CAF_RED}; border: 1px solid {CAF_RED};
}}
.stButton > button:hover[kind="primary"] {{ background:#b40b13; border-color:#b40b13; }}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <div class="logo">CAF</div>
      <div>
        <div class="title">Incidencias · Top-K</div>
        <p class="desc">Clasificación asistida y revisión con feedback del técnico</p>
      </div>
    </div>
    """, unsafe_allow_html=True
)

# ------------------- Utilidades JSON -------------------
DATA_DIR = "data"
INCIDENTS_FILE = os.path.join(DATA_DIR, "incidents.json")
PRED_FILE = os.path.join(DATA_DIR, "incidents_predichas.json")
FEEDBACK_FILE = os.path.join(DATA_DIR, "incidents_feedback.json")
REFRESH_SECONDS = 2.0

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

# --------------- Helpers robustos ---------------
def make_safe_key(item):
    _id = item.get("id")
    ts = item.get("created_at", "")
    comment = item.get("comment", "")
    return _id or f"{ts}__{abs(hash(comment))}"

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
    if score is None: return "—"
    try: return f"{100.0*float(score):.1f}%"
    except Exception: return "—"

def _status_of(it) -> str:
    status = (it.get("status") or "").lower().strip()
    if status == "ok": return "Satisfechas"
    if (it.get("revisions", []) or []): return "Revisadas"
    return "Pendientes"

# Resaltado búsqueda
import re, html
def _highlight(txt: str, q: str) -> str:
    if not q: return html.escape(txt)
    res = []; last = 0
    for m in re.finditer(re.escape(q), txt, flags=re.IGNORECASE):
        res.append(html.escape(txt[last:m.start()]))
        res.append(f"<mark class='search'>{html.escape(m.group(0))}</mark>")
        last = m.end()
    res.append(html.escape(txt[last:]))
    return ''.join(res)

# ------------------- Filtros (sidebar) -------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🔎 Filtros</div>", unsafe_allow_html=True)
    preds_all = load_predictions()

    def _to_date(s):
        try: return datetime.fromisoformat(s.replace("Z","")).date()
        except Exception: return None

    min_date = None; max_date = None
    for it in preds_all:
        d = _to_date(it.get("created_at",""))
        if d:
            min_date = d if (min_date is None or d < min_date) else min_date
            max_date = d if (max_date is None or d > max_date) else max_date
    if min_date is None: min_date = date.today()
    if max_date is None: max_date = date.today()

    start_d, end_d = st.date_input("Rango de fechas", value=(min_date, max_date))
    q_text = st.text_input("Texto en comentario", placeholder="palabras clave…").strip()

    status_options = ["— todos —", "Pendientes", "Revisadas", "Satisfechas"]
    sel_status = st.selectbox("Estado", options=status_options, index=0)

    claveros_top1 = sorted({ (it.get("prediction") or {}).get("clavero","") for it in preds_all if (it.get("prediction") or {}).get("clavero") })
    sel_clav = st.selectbox("Clavero (top-1)", options=["— todos —"] + claveros_top1, index=0)

    only_with_topk = st.checkbox("Sólo con Top-K disponible", value=False)
    sort_opt = st.selectbox("Ordenar por", options=["Fecha ↓ (recientes)", "Fecha ↑ (antiguas)", "Estado", "Clavero"], index=0)

    st.divider()
    st.toggle("Auto-refresh", key="auto_refresh", value=st.session_state["auto_refresh"])
    st.caption("Si está activo, refresca cada ~2 s (pausa si editas)")

    st.markdown("### Exportar")
    export_json_ph = st.empty()
    export_csv_ph = st.empty()

def _item_date_ok(it):
    try: d = datetime.fromisoformat(it.get("created_at","").replace("Z","")).date()
    except Exception: return False
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

def _item_topk_ok(it):
    if not only_with_topk: return True
    pred = it.get("prediction",{}) or {}
    k = ((pred.get("topk") or {}).get("claveros") or [])
    return len(k) > 0

# ------------------- KPIs -------------------
preds_all = load_predictions()
total = len(preds_all)
num_ok = sum(1 for it in preds_all if _status_of(it) == "Satisfechas")
num_rev = sum(1 for it in preds_all if _status_of(it) == "Revisadas")
num_pending = sum(1 for it in preds_all if _status_of(it) == "Pendientes")

c1, c2, c3, c4 = st.columns([1,1,1,2])
c1.metric("Total", total)
c2.metric("Pendientes", num_pending)
c3.metric("Revisadas", num_rev)
c4.metric("Satisfechas", num_ok)

st.divider()

# ------------------- Ordenado + Export -------------------
filtered = [it for it in preds_all if _item_date_ok(it) and _item_text_ok(it) and _item_clavero_ok(it) and _item_status_ok(it) and _item_topk_ok(it)]

def _sort_key(it):
    if sort_opt == "Fecha ↑ (antiguas)": return it.get("created_at","")
    if sort_opt == "Estado": return _status_of(it), it.get("created_at","")
    if sort_opt == "Clavero": return (it.get("prediction",{}) or {}).get("clavero",""), it.get("created_at","")
    return it.get("created_at","")
reverse = (sort_opt in ("Fecha ↓ (recientes)",))
filtered.sort(key=_sort_key, reverse=reverse)

try:
    export_json_ph.download_button(
        "Descargar JSON (filtro)",
        data=json.dumps(filtered, ensure_ascii=False, indent=2),
        file_name="incidencias_filtradas.json",
        mime="application/json",
        use_container_width=True
    )
    if pd is not None:
        flat = []
        for it in filtered:
            pred = it.get("prediction", {}) or {}
            flat.append({
                "id": it.get("id",""),
                "created_at": it.get("created_at",""),
                "comment": it.get("comment",""),
                "status": it.get("status",""),
                "clavero_top1": pred.get("clavero",""),
                "accion_top1": pred.get("accion",""),
                "has_topk": 1 if (((pred.get("topk") or {}).get("claveros") or [])) else 0
            })
        df_flat = pd.DataFrame(flat)
        export_csv_ph.download_button(
            "Descargar CSV (filtro)",
            data=df_flat.to_csv(index=False).encode("utf-8"),
            file_name="incidencias_filtradas.csv",
            mime="text/csv",
            use_container_width=True
        )
except Exception:
    pass

# ------------------- Render tarjeta -------------------
def _badge_html(it):
    s = _status_of(it)
    if s == "Satisfechas": return "<span class='pill badge-ok'>Satisfecha</span>"
    if s == "Revisadas":   return "<span class='pill badge-rev'>Revisada</span>"
    return "<span class='pill badge-pend'>Pendiente</span>"

def render_card(it, key_prefix: str):
    safe_key = make_safe_key(it)
    ts = it.get("created_at","—")
    comment = it.get("comment","—")
    pred = it.get("prediction",{}) or {}
    top1_clav = pred.get("clavero","—")
    top1_acc  = pred.get("accion","—")
    topk = ((pred.get("topk") or {}).get("claveros") or [])
    status = (it.get("status") or "").lower().strip()

    edit_key = f"{key_prefix}_{safe_key}"

    st.markdown('<div class="card">', unsafe_allow_html=True)

    left, right = st.columns([6,2])
    with left:
        header_html = f"<div class='card-header'><div class='meta'>{ts}</div><div>{_badge_html(it)}</div></div>"
        st.markdown(header_html, unsafe_allow_html=True)
        shown_comment = _highlight(comment, q_text) if q_text else html.escape(comment)
        st.markdown(f"<p class='kv'><b>Comentario</b>: {shown_comment}</p>", unsafe_allow_html=True)
    with right:
        pass

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<p class='kv'><b>Clavero</b>: {top1_clav}</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<p class='kv'><b>Acción</b>: {top1_acc}</p>", unsafe_allow_html=True)

    # ------- Panel Top-K / bloque cerrado -------
    if status == "ok":
        st.markdown(
            f"<div class='locked'>Clasificación cerrada. "
            f"Se fijó <b>{html.escape(top1_clav)}</b> → <b>{html.escape(top1_acc)}</b>.</div>",
            unsafe_allow_html=True
        )
    else:
        with st.expander("🔎 Ver Top-K sugerencias (3×3)", expanded=False):
            if not topk:
                calc_key = f"{key_prefix}_calc_topk_{safe_key}"
                if st.button("Calcular Top-K ahora", key=calc_key, type="primary"):
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
                # Claveros y acciones
                for ci, cblock in enumerate(topk):
                    c_label = cblock.get("label", "—")
                    st.markdown(f"**{ci+1}. {c_label}**", unsafe_allow_html=True)

                    acciones = cblock.get("acciones", []) or []
                    if not acciones:
                        st.caption("• sin acciones sugeridas")
                        continue

                    # Opciones de acciones (sin conf)
                    options = [a.get('label', '—') for a in acciones]
                    rad_key = f"{key_prefix}_rad_{safe_key}_{ci}"
                    choice = st.radio("Acciones", options=options, index=0, key=rad_key,
                                    horizontal=True, label_visibility="collapsed")

                    # 🔹 Mostrar definiciones de cada acción debajo
                    with st.container():
                        st.markdown("<div style='margin-left:1rem; margin-top:0.3rem;'>", unsafe_allow_html=True)
                        for a in acciones:
                            lbl = a.get('label', '—')
                            # 🟠 Caso especial: acción vacía o 'NAN'
                            if not lbl or str(lbl).strip().upper() == "NAN":
                                st.markdown(
                                    f"<p style='font-size:0.85rem; margin:0.1rem 0; color:#9CA3AF;'><b>{lbl}</b> — Nada que hacer</p>",
                                    unsafe_allow_html=True
                                )
                                continue

                            # 🔹 Buscar definición de esa acción
                            defi = None
                            try:
                                defi = get_def_by_clavero_accion(c_label, lbl)
                            except Exception:
                                defi = None

                            if defi:
                                # Corrección de caracteres mal codificados (acentos)
                                try:
                                    defi = defi.encode("latin1").decode("utf-8")
                                except Exception:
                                    pass
                                st.markdown(
                                    f"<p style='font-size:0.85rem; margin:0.1rem 0;'><b>{lbl}</b> — {defi}</p>",
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f"<p style='font-size:0.85rem; margin:0.1rem 0; color:#9CA3AF;'><b>{lbl}</b> — Sin definición</p>",
                                    unsafe_allow_html=True
                                )
                        st.markdown("</div>", unsafe_allow_html=True)

                    # --- botón aplicar opción ---
                    apply_key = f"{key_prefix}_apply_{safe_key}_{ci}"
                    if st.button("Aplicar esta opción", key=apply_key, help="Fijar clavero+acción, vaciar Top-K y marcar como satisfecha"):
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
                            itm["prediction"]["topk"] = {"claveros": []}
                            itm["status"] = "ok"
                            save_predictions(all_preds)

                            fb = load_feedback()
                            fb.append({
                                "id": itm.get("id"),
                                "created_at": datetime.utcnow().isoformat() + "Z",
                                "original_comment": itm.get("comment",""),
                                "original_prediction": old,
                                "user_feedback": "ok_from_topk",
                                "chosen_pair": {"clavero": new_clav, "accion": new_acc}
                            })
                            save_feedback(fb)

                            st.toast("Clasificación aplicada y cerrada ✅")
                            st.rerun()
                        else:
                            st.warning("No se pudo localizar la incidencia para actualizar.")

    # ------- Botones principales -------
    st.markdown("<div class='btn-row'>", unsafe_allow_html=True)
    ok_clicked  = st.button("✅ Satisfecho", key=f"{key_prefix}_ok_{safe_key}", disabled=(status=="ok"))
    ko_clicked  = st.button("✏️ Mejorar",  key=f"{key_prefix}_ko_{safe_key}", disabled=(status=="ok"))
    del_clicked = st.button("🗑️ Borrar pred.", key=f"{key_prefix}_del_{safe_key}")
    st.markdown("</div>", unsafe_allow_html=True)

    # OK → cerrar y limpiar Top-K
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
            all_preds[idx_match].setdefault("prediction", {}).setdefault("topk", {"claveros": []})
            all_preds[idx_match]["prediction"]["topk"] = {"claveros": []}
            save_predictions(all_preds)
            st.toast("Marcada como satisfecha ✅")
        else:
            st.warning("No se pudo localizar la incidencia para actualizar.")
        st.rerun()

    # Borrar
    if del_clicked:
        all_preds = load_predictions()
        remaining = [x for x in all_preds if not same_item(x, it)]
        save_predictions(remaining)
        st.rerun()

    # Entrar en edición
    if ko_clicked:
        st.session_state.setdefault("edit", {})[edit_key] = True

    # ------- Expander de corrección (Modelo 2) -------
    if st.session_state.get("edit", {}).get(edit_key, False):
        with st.expander("✍️ Proponer corrección y recalcular (Modelo 2)", expanded=True):
            draft_key = f"{key_prefix}_corr_{safe_key}"
            new_txt = st.text_area(
                "Describe la corrección o el contexto adicional",
                key=draft_key,
                placeholder="Ej: No es timonería; el problema real es el cableado del sensor XYZ…",
                height=120
            )
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                recalc_act = st.button("🔁 Recalcular ACTUACIONES", key=f"{key_prefix}_recalc_act_{safe_key}")
            with c2:
                recalc_all = st.button("🧠 Recalcular CLAVEROS+ACTUACIONES", key=f"{key_prefix}_recalc_all_{safe_key}")
            with c3:
                cancel_clicked = st.button("Cancelar", key=f"{key_prefix}_cancel_{safe_key}")

            if cancel_clicked:
                st.session_state.setdefault("edit", {})[edit_key] = False
                st.stop()

            if recalc_act or recalc_all:
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
                        if recalc_act:
                            claveros_exist = [c.get("label","") for c in topk] if topk else ([top1_clav] if top1_clav else [])
                            new_pred = aplicar_modelo_recalc_acciones(
                                descripcion_ot=comment,
                                comentario_feedback=correction,
                                claveros_existentes=claveros_exist,
                                top1_clavero=top1_clav,
                                k_accion=3
                            )
                        else:
                            new_pred = aplicar_modelo_recalc_todo(
                                descripcion_ot=comment,
                                comentario_feedback=correction,
                                k_clavero=3,
                                k_accion=3
                            )

                        if not isinstance(new_pred, dict) or "clavero" not in new_pred:
                            raise RuntimeError("El wrapper no devolvió una estructura válida.")

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
                            itm["prediction"] = new_pred  # claveros y/o actuaciones actualizadas
                            itm["status"] = "revised"
                            save_predictions(all_preds)
                            st.toast("Recalc completado ✅")
                        else:
                            st.warning("No se pudo localizar la incidencia para revisar.")
                    except Exception as e:
                        fb_entry["error"] = str(e)
                        st.error(f"No se pudo recalcular: {e}")

                    fb.append(fb_entry)
                    save_feedback(fb)
                    st.session_state.setdefault("edit", {})[edit_key] = True
                    st.rerun()

    # Histórico
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
if not filtered:
    st.info("No hay incidencias con los filtros actuales.")
else:
    cols = st.columns(2)
    for idx, it in enumerate(filtered):
        with cols[idx % 2]:
            render_card(it, key_prefix="main")

# ------------------- Auto-refresh (pausado si editas) -------------------
def is_any_editing() -> bool:
    return any(bool(v) for v in st.session_state.get("edit", {}).values())

if st.session_state.get("auto_refresh", True) and not is_any_editing():
    time.sleep(REFRESH_SECONDS)
    st.rerun()

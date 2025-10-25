# aplicar_modelo_wrapper.py
# -*- coding: utf-8 -*-
"""
Wrapper de aplicación de modelos CLAVERO y ACTUACIÓN.

Flujos soportados:
- aplicar_modelo_topk(descripcion_ot, k_clavero=3, k_accion=3)
    * Top-K claveros con predecir_topk_clavero / predecir_topk_clavero1  (modelo 1)
    * Para cada clavero: Top-K de actuaciones con predecir_topk_actuacion1 (modelo 1)
    * Top-1 actuación del clavero principal con Funcion_prediccion_actuacion1

- aplicar_modelo_recalc_acciones(descripcion_ot, comentario_feedback, claveros_existentes, top1_clavero, k_accion=3)
    * Mantiene el conjunto/orden de claveros
    * Recalcula SOLO las actuaciones con predecir_topk_actuacion2 (modelo 2)

- aplicar_modelo_recalc_todo(descripcion_ot, comentario_feedback, k_clavero=3, k_accion=3)
    * Recalcula Top-K de claveros con predecir_topk_clavero2 (modelo 2)
    * Para cada nuevo clavero, Top-K actuaciones con predecir_topk_actuacion2 (modelo 2)
"""

import os
import json
import traceback
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import funciones  # modelos/funciones

# ============================================================
# Resolución de ruta de modelo y carga temprana
# ============================================================
_env_path = os.environ.get("MODEL_PATH") or os.environ.get("MODELS_PATH")
_guess = str(Path(__file__).resolve().parent.parent / "models" / "modelos_baseline_clav_act.joblib")
_MODEL_PATH = _env_path if _env_path else _guess

try:
    if _MODEL_PATH:
        funciones.configurar_ruta_modelos(_MODEL_PATH)
        if os.path.exists(_MODEL_PATH):
            funciones.cargar_modelos(_MODEL_PATH)
            print(f"[OK] Modelos cargados desde {_MODEL_PATH}")
        else:
            print(f"[WARN] Modelo no encontrado en {_MODEL_PATH}")
    else:
        print("[WARN] MODEL_PATH/MODELS_PATH no definidos y fallback vacío")
except Exception as e:
    print(f"[WARN] No se pudieron cargar modelos desde {_MODEL_PATH}: {e}")


# ============================================================
# Auxiliares
# ============================================================
def _safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[WARN] Error en {getattr(fn, '__name__', str(fn))}: {e}")
        traceback.print_exc()
        return None

def _normalize_pairs(obj: Any, k: Optional[int]) -> List[Tuple[str, Optional[float]]]:
    """
    Normaliza top-k -> [(label, score?)] desde list[str], list[(str,score)] o dict[str,score].
    """
    if obj is None:
        return []
    out: List[Tuple[str, Optional[float]]] = []

    if isinstance(obj, dict):
        out = list(obj.items())
    elif hasattr(obj, "items"):
        try:
            out = list(obj.items())  # type: ignore
        except Exception:
            pass
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            if isinstance(x, (list, tuple)) and len(x) >= 1:
                lbl = str(x[0])
                sc = float(x[1]) if len(x) >= 2 and isinstance(x[1], (int, float)) else None
                out.append((lbl, sc))
            else:
                out.append((str(x), None))
    else:
        out = [(str(obj), None)]

    # ordena por score si hubiera
    if out and any(s is not None for _, s in out):
        out = sorted(out, key=lambda t: (t[1] is not None, t[1]), reverse=True)  # type: ignore
    if k is not None and k > 0:
        out = out[:k]
    return out


# ============================================================
# Top-K Clavero (pase inicial, modelo 1)
# ============================================================
def _call_best_effort_topk_clavero(comentario: str, k: int) -> List[Tuple[str, Optional[float]]]:
    # 1) predecir_topk_clavero (alias recomendado)
    for name in ("predecir_topk_clavero", "predecir_topk_clavero1"):
        fn = getattr(funciones, name, None)
        if callable(fn):
            try:
                res = fn(comentario, k=k)
                return _normalize_pairs(res, k)
            except Exception:
                traceback.print_exc()

    # Legacy
    for name in ("Funcion_prediccion_clavero_topk", "Funcion_prediccion_clavero_probas"):
        fn = getattr(funciones, name, None)
        if callable(fn):
            res = _safe_call(fn, comentario, k=k)
            if res is not None:
                return _normalize_pairs(res, k)
    return []


# ============================================================
# Top-K Clavero con feedback (modelo 2)
# ============================================================
def _call_best_effort_topk_clavero2(comentario: str, feedback: str, k: int) -> List[Tuple[str, Optional[float]]]:
    fn = getattr(funciones, "predecir_topk_clavero2", None)
    if callable(fn):
        # Firma (comentario, feedback, k=3)
        try:
            res = fn(comentario, feedback, k=k)
            return _normalize_pairs(res, k)
        except TypeError:
            # Firma (texto_unido, k=3)
            try:
                res = fn(f"{comentario}\nCorrección: {feedback}", k=k)
                return _normalize_pairs(res, k)
            except Exception:
                traceback.print_exc()
        except Exception:
            traceback.print_exc()
    return []


# ============================================================
# Top-K Actuación
# ============================================================
def _call_best_effort_topk_accion(clavero: str, comentario: str, k: int) -> List[Tuple[str, Optional[float]]]:
    # Prioriza modelo 1 (sin feedback)
    k=1
    for name in ("predecir_topk_actuacion1", "predecir_topk_actuacion2"):
        fn = getattr(funciones, name, None)
        if callable(fn):
            try:
                if name.endswith("2"):
                    res = fn(clavero, comentario, None, None, k=k)
                else:
                    res = fn(clavero, comentario, k=k)
                return _normalize_pairs(res, k)
            except Exception:
                traceback.print_exc()

    # Legacy
    for name in ("Funcion_prediccion_actuacion_topk", "Funcion_prediccion_actuacion_probas"):
        fn = getattr(funciones, name, None)
        if callable(fn):
            res = _safe_call(fn, clavero, comentario, k=k)
            if res is not None:
                return _normalize_pairs(res, k)
    return []


def _call_best_effort_topk_accion2(clavero: str, comentario: str, feedback: str, k: int) -> List[Tuple[str, Optional[float]]]:
    k=1
    fn = getattr(funciones, "predecir_topk_actuacion2", None)
    if callable(fn):
        try:
            res = fn(clavero, comentario, feedback, None, k=k)
            return _normalize_pairs(res, k)
        except Exception:
            traceback.print_exc()
    return []


# ============================================================
# API pública del wrapper
# ============================================================
def aplicar_modelo(descripcion_ot: str) -> Dict[str, Any]:
    """Top-1 simple (compat)."""
    fn_clav = getattr(funciones, "Funcion_prediccion_clavero", None)
    if not callable(fn_clav):
        raise RuntimeError("No se encontró Funcion_prediccion_clavero en funciones.py")

    clavero_pred = str(_safe_call(fn_clav, descripcion_ot) or "")

    fn_act1 = getattr(funciones, "Funcion_prediccion_actuacion1", None)
    if callable(fn_act1):
        actuacion_pred = str(_safe_call(fn_act1, clavero_pred, descripcion_ot) or "")
    else:
        fn_act2 = getattr(funciones, "Funcion_prediccion_actuacion2", None)
        actuacion_pred = str(_safe_call(fn_act2, clavero_pred, descripcion_ot, None, None) or "") if callable(fn_act2) else ""

    return {"clavero": clavero_pred, "accion": actuacion_pred}


def aplicar_modelo_topk(descripcion_ot: str, k_clavero: int = 3, k_accion: int = 3) -> Dict[str, Any]:
    """Pase inicial (modelo 1)."""
    # Top-1 clavero
    fn_clav = getattr(funciones, "Funcion_prediccion_clavero", None)
    if not callable(fn_clav):
        raise RuntimeError("No se encontró Funcion_prediccion_clavero en funciones.py")
    clavero_pred = str(_safe_call(fn_clav, descripcion_ot) or "")

    # Top-K claveros (modelo 1)
    topk_claveros = _call_best_effort_topk_clavero(descripcion_ot, k=k_clavero)

    # Top-K actuaciones por cada clavero (modelo 1)
    claveros_struct = []
    for clav_label, clav_score in topk_claveros:
        acciones = _call_best_effort_topk_accion(clav_label, descripcion_ot, k=k_accion)
        claveros_struct.append({
            "label": clav_label,
            "score": clav_score,
            "acciones": [{"label": a, "score": s} for a, s in acciones]
        })

    # Top-1 actuación (modelo 1)
    fn_act1 = getattr(funciones, "Funcion_prediccion_actuacion1", None)
    if callable(fn_act1):
        actuacion_pred = str(_safe_call(fn_act1, clavero_pred, descripcion_ot) or "")
    else:
        fn_act2 = getattr(funciones, "Funcion_prediccion_actuacion2", None)
        actuacion_pred = str(_safe_call(fn_act2, clavero_pred, descripcion_ot, None, None) or "") if callable(fn_act2) else ""

    return {"clavero": clavero_pred, "accion": actuacion_pred, "topk": {"claveros": claveros_struct}}


def aplicar_modelo_recalc_acciones(
    descripcion_ot: str,
    comentario_feedback: str,
    claveros_existentes: List[str],
    top1_clavero: str,
    k_accion: int = 3,
) -> Dict[str, Any]:
    """Recalcula SOLO actuaciones con modelo 2, manteniendo claveros."""
    claveros_existentes = [str(c).strip() for c in (claveros_existentes or []) if str(c).strip()]
    if not claveros_existentes:
        if top1_clavero:
            claveros_existentes = [top1_clavero]
        else:
            return {"clavero": "", "accion": "", "topk": {"claveros": []}}

    claveros_struct = []
    for label in claveros_existentes:
        acts = _call_best_effort_topk_accion2(label, descripcion_ot, comentario_feedback, k=k_accion)
        claveros_struct.append({
            "label": label,
            "score": None,
            "acciones": [{"label": a, "score": s} for a, s in acts]
        })

    try:
        acts_top1 = _call_best_effort_topk_accion2(top1_clavero, descripcion_ot, comentario_feedback, k=1)
        accion_top1 = acts_top1[0][0] if acts_top1 else ""
    except Exception:
        accion_top1 = ""

    return {"clavero": top1_clavero, "accion": accion_top1, "topk": {"claveros": claveros_struct}}


def aplicar_modelo_recalc_todo(
    descripcion_ot: str,
    comentario_feedback: str,
    k_clavero: int = 3,
    k_accion: int = 3,
) -> Dict[str, Any]:
    """Recalcula claveros (modelo 2) + actuaciones (modelo 2) con el feedback."""
    # Nuevo Top-K claveros
    topk_claveros = _call_best_effort_topk_clavero2(descripcion_ot, comentario_feedback, k=k_clavero)

    claveros_struct = []
    for clav_label, clav_score in topk_claveros:
        acts = _call_best_effort_topk_accion2(clav_label, descripcion_ot, comentario_feedback, k=k_accion)
        claveros_struct.append({
            "label": clav_label,
            "score": clav_score,
            "acciones": [{"label": a, "score": s} for a, s in acts]
        })

    top1_clav = topk_claveros[0][0] if topk_claveros else ""
    if top1_clav:
        acts_top1 = _call_best_effort_topk_accion2(top1_clav, descripcion_ot, comentario_feedback, k=1)
        accion_top1 = acts_top1[0][0] if acts_top1 else ""
    else:
        accion_top1 = ""

    return {"clavero": top1_clav, "accion": accion_top1, "topk": {"claveros": claveros_struct}}

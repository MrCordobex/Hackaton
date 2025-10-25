# aplicar_modelo_wrapper.py
# -*- coding: utf-8 -*-
"""
Wrapper de aplicación de modelos CLAVERO y ACTUACIÓN.
Flujo:
- aplicar_modelo_topk(descripcion_ot, k_clavero=3, k_accion=3):
    * Top-K de claveros con predecir_topk_clavero
    * Para cada clavero: Top-K de actuaciones con predecir_topk_actuacion1 (modelo 1)
    * Top-1 actuación para el clavero principal con Funcion_prediccion_actuacion1
- aplicar_modelo_recalc_acciones(descripcion_ot, comentario_feedback, claveros_existentes, top1_clavero, k_accion=3):
    * Mantiene el conjunto/orden de claveros (NO se recalculan)
    * Recalcula SOLO las actuaciones con predecir_topk_actuacion2 (modelo 2, con feedback)
"""

import os
import json
import traceback
from typing import Dict, List, Tuple, Any
from pathlib import Path
import funciones  # importa una sola vez


# ============================================================
# Resolución y carga temprana del modelo
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


# ============================================================
# Top-K Clavero
# ============================================================
def _call_best_effort_topk_clavero(comentario: str, k: int) -> List[Tuple[str, float]]:
    # 1) Tu función real
    fn = getattr(funciones, "predecir_topk_clavero", None)
    if callable(fn):
        try:
            labels = fn(comentario, k=k)
            return [(str(l), None) for l in labels][:k]
        except Exception:
            traceback.print_exc()

    # 2) Compatibilidad legacy
    for name in ("Funcion_prediccion_clavero_topk", "Funcion_prediccion_clavero_probas"):
        fn = getattr(funciones, name, None)
        if callable(fn):
            res = _safe_call(fn, comentario, k=k)
            if res is not None:
                if isinstance(res, list):
                    if all(isinstance(r, tuple) and len(r) == 2 for r in res):
                        return res[:k]
                    else:
                        return [(str(r), None) for r in res][:k]
    return []


# ============================================================
# Top-K Actuación (flujo inicial: modelo 1)
# ============================================================
def _call_best_effort_topk_accion(clavero: str, comentario: str, k: int) -> List[Tuple[str, float]]:
    # Prioriza el modelo 1 en el pase inicial
    for name in ("predecir_topk_actuacion1", "predecir_topk_actuacion2"):
        fn = getattr(funciones, name, None)
        if callable(fn):
            try:
                if name.endswith("2"):
                    labels = fn(clavero, comentario, None, None, k=k)
                else:
                    labels = fn(clavero, comentario, k=k)
                return [(str(l), None) for l in labels][:k]
            except Exception:
                traceback.print_exc()

    # Compatibilidad legacy
    for name in ("Funcion_prediccion_actuacion_topk", "Funcion_prediccion_actuacion_probas"):
        fn = getattr(funciones, name, None)
        if callable(fn):
            res = _safe_call(fn, clavero, comentario, k=k)
            if res is not None:
                if isinstance(res, list):
                    if all(isinstance(r, tuple) and len(r) == 2 for r in res):
                        return res[:k]
                    else:
                        return [(str(r), None) for r in res][:k]
    return []


# ============================================================
# API pública del wrapper
# ============================================================
def aplicar_modelo(descripcion_ot: str) -> Dict[str, Any]:
    """
    Top-1 (compatibilidad).
    Estructura mínima: { "clavero": <str>, "accion": <str> }
    """
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


def aplicar_modelo_topk(
    descripcion_ot: str,
    k_clavero: int = 3,
    k_accion: int = 3,
) -> Dict[str, Any]:
    """
    Top-K para visualización (pase inicial con modelo 1 para actuaciones).
    """
    # Top-1 clavero
    fn_clav = getattr(funciones, "Funcion_prediccion_clavero", None)
    if not callable(fn_clav):
        raise RuntimeError("No se encontró Funcion_prediccion_clavero en funciones.py")
    clavero_pred = str(_safe_call(fn_clav, descripcion_ot) or "")

    # Top-K claveros
    topk_claveros = _call_best_effort_topk_clavero(descripcion_ot, k=k_clavero)

    # Top-K acciones por cada clavero (modelo 1)
    claveros_struct = []
    for clav_label, clav_score in topk_claveros:
        acciones = _call_best_effort_topk_accion(clav_label, descripcion_ot, k=k_accion)
        claveros_struct.append({
            "label": clav_label,
            "score": clav_score,
            "acciones": [{"label": a, "score": s} for a, s in acciones]
        })

    # Top-1 actuación para el clavero principal (modelo 1)
    fn_act1 = getattr(funciones, "Funcion_prediccion_actuacion1", None)
    if callable(fn_act1):
        actuacion_pred = str(_safe_call(fn_act1, clavero_pred, descripcion_ot) or "")
    else:
        fn_act2 = getattr(funciones, "Funcion_prediccion_actuacion2", None)
        actuacion_pred = str(_safe_call(fn_act2, clavero_pred, descripcion_ot, None, None) or "") if callable(fn_act2) else ""

    return {
        "clavero": clavero_pred,
        "accion": actuacion_pred,
        "topk": {"claveros": claveros_struct}
    }


def aplicar_modelo_recalc_acciones(
    descripcion_ot: str,
    comentario_feedback: str,
    claveros_existentes: List[str],
    top1_clavero: str,
    k_accion: int = 3,
) -> Dict[str, Any]:
    """
    Recalcula SOLO las actuaciones usando el modelo 2 (con feedback),
    manteniendo el conjunto/orden de claveros.
    """
    # Normaliza lista de claveros existente
    claveros_existentes = [str(c).strip() for c in (claveros_existentes or []) if str(c).strip()]
    if not claveros_existentes:
        if top1_clavero:
            claveros_existentes = [top1_clavero]
        else:
            return {"clavero": "", "accion": "", "topk": {"claveros": []}}

    # Recalcular Top-K de actuaciones por cada clavero con modelo 2
    claveros_struct = []
    for label in claveros_existentes:
        try:
            acts = funciones.predecir_topk_actuacion2(label, descripcion_ot, comentario_feedback, None, k=k_accion)
        except Exception:
            traceback.print_exc()
            acts = []
        claveros_struct.append({
            "label": label,
            "score": None,
            "acciones": [{"label": a, "score": None} for a in acts]
        })

    # Nueva top-1 actuación para el clavero principal
    try:
        acts_top1 = funciones.predecir_topk_actuacion2(top1_clavero, descripcion_ot, comentario_feedback, None, k=1)
        accion_top1 = acts_top1[0] if acts_top1 else ""
    except Exception:
        traceback.print_exc()
        accion_top1 = ""

    return {
        "clavero": top1_clavero,
        "accion": accion_top1,
        "topk": {"claveros": claveros_struct}
    }


if __name__ == "__main__":
    texto = input("Descripción de la OT: ")
    res = aplicar_modelo_topk(texto, k_clavero=3, k_accion=3)
    print(json.dumps(res, indent=2, ensure_ascii=False))

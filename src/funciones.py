# funciones.py
# -*- coding: utf-8 -*-
"""
Módulo de utilidades para:
 - Preprocesado y vectorización de texto (palabra 1-3 + carácter 3-6, stopwords ES)
 - Entrenamiento de modelos CLAVERO y ACTUACIÓN (SVC balanced + calibración robusta)
 - Predicción:
     * Funcion_prediccion_clavero(descripcion_ot) -> clavero (str)
     * Funcion_prediccion_actuacion1(clavero, descripcion_ot) -> actuación (str)
     * Funcion_prediccion_actuacion2(clavero, descripcion_ot, descripcion_averia, comentarios) -> actuación (str)
 - Funcion_texto(textos, modo='clavero', clavero_context=None) -> matriz TF-IDF (hstack palabra+carácter)

NOTAS:
 - Llama primero a entrenar_modelos(...) o a cargar_modelos(...) antes de predecir.
 - 'NAN' en actuación es clase válida (no se filtra).
"""

import re
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import joblib

import os
from pathlib import Path

_DEFAULT_ARTIFACTS_PATH = os.environ.get(
    "MODELS_PATH",
    str(Path(__file__).with_name("modelos.joblib"))
)

def configurar_ruta_modelos(path: str) -> None:
    """Fija la ruta del .joblib sin cargar aún (opcional)."""
    global _DEFAULT_ARTIFACTS_PATH
    _DEFAULT_ARTIFACTS_PATH = path

def _ensure_models_loaded():
    """Carga una vez si hace falta (lazy)."""
    global _clf_clav, _vect_word_clav, _vect_char_clav, _clav_classes
    global _clf_act,  _vect_word_act,  _vect_char_act,  _act_classes

    need_clav = any(x is None for x in (_clf_clav, _vect_word_clav, _vect_char_clav, _clav_classes))
    need_act  = any(x is None for x in (_clf_act,  _vect_word_act,  _vect_char_act,  _act_classes))

    if need_clav or need_act:
        if not _DEFAULT_ARTIFACTS_PATH or not os.path.exists(_DEFAULT_ARTIFACTS_PATH):
            raise RuntimeError(
                "No hay modelos en memoria y no se encuentra el archivo de artefactos.\n"
                "Soluciones: 1) ejecuta entrenar_modelos(...) y guardar_modelos('modelos.joblib'), "
                "2) coloca 'modelos.joblib' junto a funciones.py, o "
                "3) llama a configurar_ruta_modelos('ruta/al/joblib')."
            )
        cargar_modelos(_DEFAULT_ARTIFACTS_PATH)


# -------------------------------
# Stopwords (NLTK)
# -------------------------------
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
SPANISH_STOPWORDS = stopwords.words('spanish')

# -------------------------------
# Limpieza / normalización
# -------------------------------

_JERGA_PATTERNS = [
    (r"\bf\.?e\.?\b", " freno_estacionamiento "),
    (r"\btdp\b", " tdp "),
    (r"\bta[_\- ]?eq1\b", " ta_eq1 "),
    (r"\bev[fs]\b", " evf "),
    (r"\banti?retorno\b", " antirretorno "),
    (r"\bvarilla\s+nivelaci[oó]n\b", " varilla_nivelacion "),
]

_HTML_TOKENS = r"\b(html|body|div|span|br|href|style|class|id)\b"

def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"&[a-z]+;", " ", s)
    s = re.sub(_HTML_TOKENS, " ", s)
    return s

def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.lower()
    # urls / emails
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", s)
    # html / dominios app
    s = _strip_html(s)
    s = re.sub(r"\b(rosmiman|leadmind|cafdigitalservices|caf)\b", " ", s)
    # acentos
    s = (s.replace("á","a").replace("é","e").replace("í","i")
           .replace("ó","o").replace("ú","u").replace("ü","u").replace("ñ","n"))
    # abreviaturas
    s = re.sub(r"\bdcha\b", " derecha ", s)
    s = re.sub(r"\bizq\b", " izquierda ", s)
    # codigos UT/OT y OTs
    s = re.sub(r"\but[_\- ]?\d+\b", " ", s)
    s = re.sub(r"\bot[_\-]?[a-z0-9_]*\b", " ", s)
    s = re.sub(r"\borden\s*trabajo\b", " ", s)
    # fechas, años, horas
    s = re.sub(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", " ", s)
    s = re.sub(r"\b\d{4}\b", " ", s)
    s = re.sub(r"\b\d{1,2}:\d{2}\b", " ", s)
    # normalizaciones jerga
    s = re.sub(r"[_\-]+", " ", s)
    for pat, rep in _JERGA_PATTERNS:
        s = re.sub(pat, rep, s)
    # limpieza final
    s = re.sub(r"[^a-z0-9\s\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -------------------------------
# Utilidades calibración
# -------------------------------

def _min_class_count(y: np.ndarray) -> int:
    _, counts = np.unique(y, return_counts=True)
    return int(np.min(counts)) if len(counts) else 0

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

class _SoftmaxSVCWrapper:
    """Envuelve un LinearSVC y expone predict_proba vía decision_function -> softmax."""
    def __init__(self, estimator: LinearSVC):
        self.estimator = estimator
        self.classes_ = None
    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_
        return self
    def predict_proba(self, X):
        d = self.estimator.decision_function(X)
        if d.ndim == 1:  # binario
            d = np.vstack([-d, d]).T
        return _softmax(d, axis=1)
    def predict(self, X):
        return self.estimator.predict(X)

def _calibrated_or_softmax(estimator: LinearSVC, X, y):
    """Devuelve un modelo con .predict_proba calibrado si es posible; si no, softmax wrapper."""
    mcc = _min_class_count(y)
    if mcc >= 3:
        model = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=3)
        model.fit(X, y)
        return model
    if mcc == 2:
        model = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=2)
        model.fit(X, y)
        return model
    model = _SoftmaxSVCWrapper(estimator=estimator)
    model.fit(X, y)
    return model

# -------------------------------
# Estado global (modelos y vectores)
# -------------------------------

# CLAVERO
_vect_word_clav: Optional[TfidfVectorizer] = None
_vect_char_clav: Optional[TfidfVectorizer] = None
_clf_clav = None
_clav_classes: Optional[np.ndarray] = None

# ACTUACION
_vect_word_act: Optional[TfidfVectorizer] = None
_vect_char_act: Optional[TfidfVectorizer] = None
_clf_act = None
_act_classes: Optional[np.ndarray] = None

# -------------------------------
# Vectorización
# -------------------------------

def _make_vectorizers(modo: str):
    """Crea los vectorizadores de palabra/caracter según modo ('clavero'|'actuacion')."""
    if modo not in {"clavero", "actuacion"}:
        raise ValueError("modo debe ser 'clavero' o 'actuacion'")
    vw = TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df=2, stop_words=SPANISH_STOPWORDS)
    vc = TfidfVectorizer(analyzer="char", ngram_range=(3,6), min_df=2)
    return vw, vc

def _fit_union(texts: List[str], vw: TfidfVectorizer, vc: TfidfVectorizer) -> csr_matrix:
    Xw = vw.fit_transform(texts)
    Xc = vc.fit_transform(texts)
    return hstack([Xw, Xc]).tocsr()

def _tr_union(texts: List[str], vw: TfidfVectorizer, vc: TfidfVectorizer) -> csr_matrix:
    Xw = vw.transform(texts)
    Xc = vc.transform(texts)
    return hstack([Xw, Xc]).tocsr()

def _concat_text(descripcion_ot: str,
                 descripcion_averia: Optional[str] = None,
                 comentarios: Optional[str] = None) -> str:
    parts = [descripcion_ot or "", descripcion_averia or "", comentarios or ""]
    return " . ".join([_normalize_text(p) for p in parts if p is not None])

# ==========================================================
# API PÚBLICA
# ==========================================================

def entrenar_modelos(
    desc_ot: List[str],
    desc_averia: Optional[List[str]],
    comentarios: Optional[List[str]],
    y_clavero: List[str],
    y_actuacion: List[str]
) -> Dict[str, float]:
    """
    Entrena los modelos globales de CLAVERO y ACTUACIÓN.
    - y_actuacion debe mantener 'NAN' si es clase válida.
    Devuelve métricas básicas sobre el propio set (hold-in).
    """
    global _vect_word_clav, _vect_char_clav, _clf_clav, _clav_classes
    global _vect_word_act, _vect_char_act, _clf_act, _act_classes

    # Texto base
    textos = [
        _concat_text(ot, (desc_averia[i] if desc_averia is not None else None),
                          (comentarios[i] if comentarios is not None else None))
        for i, ot in enumerate(desc_ot)
    ]

    y_clav = pd.Series(y_clavero, dtype=str).str.upper().str.strip().tolist()
    y_act  = pd.Series(y_actuacion, dtype=str).fillna("NAN").str.upper().str.strip().tolist()

    # ----- CLAVERO -----
    _vect_word_clav, _vect_char_clav = _make_vectorizers("clavero")
    X_clav = _fit_union(textos, _vect_word_clav, _vect_char_clav)
    base_clav = LinearSVC(class_weight="balanced")
    _clf_clav = _calibrated_or_softmax(base_clav, X_clav, y_clav)
    _clav_classes = _clf_clav.classes_

    yhat_c = _clav_classes[_clf_clav.predict_proba(X_clav).argmax(axis=1)]
    acc_c = accuracy_score(y_clav, yhat_c)
    f1m_c = f1_score(y_clav, yhat_c, average="macro")

    # ----- ACTUACIÓN -----
    def _add_token(texts: List[str], claveros: List[str]) -> List[str]:
        return [f"__clav_{c}__ {t}" for c, t in zip(claveros, texts)]

    # En train, usar clavero REAL como contexto
    textos_aug = _add_token(textos, y_clav)

    _vect_word_act, _vect_char_act = _make_vectorizers("actuacion")
    X_act = _fit_union(textos_aug, _vect_word_act, _vect_char_act)
    base_act = LinearSVC(class_weight="balanced")
    _clf_act = _calibrated_or_softmax(base_act, X_act, y_act)
    _act_classes = _clf_act.classes_

    yhat_a = _act_classes[_clf_act.predict_proba(X_act).argmax(axis=1)]
    acc_a = accuracy_score(y_act, yhat_a)
    f1m_a = f1_score(y_act, yhat_a, average="macro")

    return {
        "clavero_acc": float(acc_c),
        "clavero_f1_macro": float(f1m_c),
        "actuacion_acc": float(acc_a),
        "actuacion_f1_macro": float(f1m_a),
    }

def guardar_modelos(path: str) -> None:
    """Guarda en disco los vectores y clasificadores."""
    artifacts = {
        "vect_word_clav": _vect_word_clav,
        "vect_char_clav": _vect_char_clav,
        "clf_clav": _clf_clav,
        "clav_classes": _clav_classes,
        "vect_word_act": _vect_word_act,
        "vect_char_act": _vect_char_act,
        "clf_act": _clf_act,
        "act_classes": _act_classes,
    }
    joblib.dump(artifacts, path)

def cargar_modelos(path: str) -> None:
    """Carga desde disco los vectores y clasificadores."""
    global _vect_word_clav, _vect_char_clav, _clf_clav, _clav_classes
    global _vect_word_act, _vect_char_act, _clf_act, _act_classes
    artifacts = joblib.load(path)
    _vect_word_clav = artifacts["vect_word_clav"]
    _vect_char_clav = artifacts["vect_char_clav"]
    _clf_clav       = artifacts["clf_clav"]
    _clav_classes   = artifacts["clav_classes"]
    _vect_word_act  = artifacts["vect_word_act"]
    _vect_char_act  = artifacts["vect_char_act"]
    _clf_act        = artifacts["clf_act"]
    _act_classes    = artifacts["act_classes"]

# ----------------------------------------------------------
# FUNCIONES SOLICITADAS
# ----------------------------------------------------------

def Funcion_texto(textos: List[str],
                  modo: str = "clavero",
                  clavero_context: Optional[List[str]] = None) -> csr_matrix:
    """
    Vectoriza una columna/lista de textos con el mismo pipeline del entrenamiento.
    - modo: 'clavero' | 'actuacion'
    - clavero_context: si modo='actuacion', lista de claveros para anteponer como token.
    """
    assert modo in {"clavero", "actuacion"}, "modo debe ser 'clavero' o 'actuacion'"

    # normalizar
    textos_norm = [_normalize_text(t) for t in textos]

    if modo == "clavero":
        if _vect_word_clav is None or _vect_char_clav is None:
            raise RuntimeError("Modelos de CLAVERO no entrenados/cargados.")
        return _tr_union(textos_norm, _vect_word_clav, _vect_char_clav)

    # modo actuacion
    if _vect_word_act is None or _vect_char_act is None:
        raise RuntimeError("Modelos de ACTUACIÓN no entrenados/cargados.")
    if clavero_context is None:
        raise ValueError("Para 'actuacion' debes proporcionar clavero_context (lista del mismo tamaño).")
    textos_aug = [f"__clav_{c}__ {t}" for c, t in zip(clavero_context, textos_norm)]
    return _tr_union(textos_aug, _vect_word_act, _vect_char_act)

def Funcion_prediccion_clavero(descripcion_ot: str) -> str:
    """
    Recibe SOLO descripcion_ot y devuelve el CLAVERO (string) top-1.
    """
    _ensure_models_loaded()
    if _clf_clav is None:
        raise RuntimeError("Modelo de CLAVERO no entrenado/cargado.")
    txt = _normalize_text(descripcion_ot or "")
    X = _tr_union([txt], _vect_word_clav, _vect_char_clav)
    proba = _clf_clav.predict_proba(X)[0]
    return _clav_classes[int(np.argmax(proba))]

def Funcion_prediccion_actuacion1(clavero: str, descripcion_ot: str) -> str:
    """
    Recibe clavero + SOLO descripcion_ot y devuelve la ACTUACIÓN (string) top-1.
    """
    _ensure_models_loaded()
    if _clf_act is None:
        raise RuntimeError("Modelo de ACTUACIÓN no entrenado/cargado.")
    txt = _normalize_text(descripcion_ot or "")
    txt_aug = f"__clav_{(clavero or '').strip().upper()}__ {txt}"
    X = _tr_union([txt_aug], _vect_word_act, _vect_char_act)
    proba = _clf_act.predict_proba(X)[0]
    return _act_classes[int(np.argmax(proba))]

def Funcion_prediccion_actuacion2(
    clavero: str,
    descripcion_ot: str,
    descripcion_averia: Optional[str],
    comentarios: Optional[str]
) -> str:
    """
    Recibe clavero + (descripcion_ot, descripcion_averia, comentarios) y devuelve ACTUACIÓN top-1.
    """
    _ensure_models_loaded()
    if _clf_act is None:
        raise RuntimeError("Modelo de ACTUACIÓN no entrenado/cargado.")
    txt = _concat_text(descripcion_ot or "", descripcion_averia, comentarios)
    txt_aug = f"__clav_{(clavero or '').strip().upper()}__ {txt}"
    X = _tr_union([txt_aug], _vect_word_act, _vect_char_act)
    proba = _clf_act.predict_proba(X)[0]
    return _act_classes[int(np.argmax(proba))]

# ----------------------------------------------------------
# Helpers opcionales (no requeridos, pero útiles)
# ----------------------------------------------------------

def predecir_topk_clavero(descripcion_ot: str, k: int = 3) -> List[str]:
    """Devuelve top-k etiquetas CLAVERO (sin probs), únicas y ordenadas por prob desc."""
    _ensure_models_loaded()
    if _clf_clav is None:
        raise RuntimeError("Modelo de CLAVERO no entrenado/cargado.")
    txt = _normalize_text(descripcion_ot or "")
    X = _tr_union([txt], _vect_word_clav, _vect_char_clav)
    p = _clf_clav.predict_proba(X)[0]
    idx_sorted = np.argsort(p)[::-1]
    out, seen = [], set()
    for i in idx_sorted:
        lab = str(_clav_classes[i])
        if lab not in seen:
            out.append(lab)
            seen.add(lab)
            if len(out) == k:
                break
    return out

def predecir_topk_actuacion2(clavero: str,
                            descripcion_ot: str,
                            descripcion_averia: Optional[str] = None,
                            comentarios: Optional[str] = None,
                            k: int = 3) -> List[str]:
    """Top-k de ACTUACIÓN (con texto ampliado) como lista de etiquetas únicas (sin probs)."""
    _ensure_models_loaded()
    if _clf_act is None:
        raise RuntimeError("Modelo de ACTUACIÓN no entrenado/cargado.")
    txt = _concat_text(descripcion_ot or "", descripcion_averia, comentarios)
    txt_aug = f"__clav_{(clavero or '').strip().upper()}__ {txt}"
    X = _tr_union([txt_aug], _vect_word_act, _vect_char_act)
    p = _clf_act.predict_proba(X)[0]
    idx_sorted = np.argsort(p)[::-1]
    out, seen = [], set()
    for i in idx_sorted:
        lab = str(_act_classes[i])
        if lab not in seen:
            out.append(lab)
            seen.add(lab)
            if len(out) == k:
                break
    return out


def predecir_topk_actuacion1(clavero: str,
                             descripcion_ot: str,
                             k: int = 3) -> List[str]:
    """Top-k de ACTUACIÓN usando SOLO clavero + descripcion_ot. Etiquetas únicas (sin probs)."""
    _ensure_models_loaded()
    if _clf_act is None:
        raise RuntimeError("Modelo de ACTUACIÓN no entrenado/cargado.")
    txt = _normalize_text(descripcion_ot or "")
    txt_aug = f"__clav_{(clavero or '').strip().upper()}__ {txt}"
    X = _tr_union([txt_aug], _vect_word_act, _vect_char_act)
    p = _clf_act.predict_proba(X)[0]
    idx_sorted = np.argsort(p)[::-1]
    out, seen = [], set()
    for i in idx_sorted:
        lab = str(_act_classes[i])
        if lab not in seen:
            out.append(lab)
            seen.add(lab)
            if len(out) == k:
                break
    return out

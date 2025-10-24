# limpiar_descripcion.py
import re
from collections import Counter
import pandas as pd

def limpiar_descripcion(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Limpia y estructura la columna de descripciones del dataset original.

    Pasos que realiza:
      1) Reemplazos globales en TODO el DataFrame:
         - "Ã‘"->"Ñ", "Ãñ"->"ñ"
         - "Ã³"->"o",  'Ã"'->"O"
         - Elimina caracteres raros: º, ª, Ã
      2) Divide la columna `col_name` en dos partes por la PRIMERA aparición
         de 4 letras consecutivas, SALTANDO si esa cuaterna es "ROSM".
         -> primera parte = 'ubicacion', segunda parte = 'descripcion'
      3) Extrae el primer patrón tipo "UT" + 3 dígitos (con o sin espacios) de la descripción.
      4) Encuentra tokens alfanuméricos (>=4, con letras y dígitos, admite _ y -) que se repiten
         en el dataset (frecuencia >= 2) y crea 'codigos_repetidos' por fila.
      5) Borra de 'descripcion' la primera ocurrencia del patrón "UT + 3 dígitos"
         (formato flexible) y todos los 'codigos_repetidos' detectados.
      6) Devuelve SOLO 3 columnas: ['ubicacion', 'descripcion', 'codigos_repetidos'].
         Si la fila original está vacía (NaN en `col_name`), se deja como None en las 3 columnas.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original ya cargado.
    col_name : str
        Nombre de la columna a limpiar.

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas: ['ubicacion', 'descripcion', 'codigos_repetidos'].
    """
    if col_name not in df.columns:
        raise KeyError(f"La columna '{col_name}' no existe en el DataFrame.")

    # --- Copia para no mutar df original ---
    _df = df.copy()

    # --- (1) Reemplazos globales en TODO el dataset ---
    def _fix_cell_global(x):
        if pd.isna(x):
            return x
        s = str(x)
        # Primero Ñ/ñ
        s = s.replace("Ã‘", "Ñ").replace("Ã±", "ñ")
        # Luego ó / O
        s = s.replace("Ã³", "o").replace('Ã"', "O")
        # Eliminar caracteres raros
        s = re.sub(r"[ºªÃ]", "", s)
        # Espacios
        s = re.sub(r"\s+", " ", s).strip()
        return s

    _df = _df.applymap(_fix_cell_global)

    # --- (2) Split en 4 letras consecutivas (saltando ROSM) ---
    pat_4letters = re.compile(r"[A-Za-z]{4}")

    def _split_first4_skip_rosm(txt):
        if pd.isna(txt):
            return None, None
        s = str(txt)
        matches = list(pat_4letters.finditer(s))
        if not matches:
            # No hay 4 letras seguidas -> todo queda como 'ubicacion', descripcion vacía
            return s.strip(), ""
        # Elegir la primera cuaterna que NO sea ROSM
        chosen = None
        for m in matches:
            if m.group(0).upper() != "ROSM":
                chosen = m
                break
        if chosen is None:
            chosen = matches[0]
        i = chosen.start()
        return s[:i].strip(), s[i:].strip()

    ubicacion_list = []
    descripcion_list = []
    for val in _df[col_name]:
        ub, desc = _split_first4_skip_rosm(val)
        ubicacion_list.append(ub)
        descripcion_list.append(desc)

    # Armamos un df intermedio con solo lo que nos interesa
    out = pd.DataFrame({
        "ubicacion": ubicacion_list,
        "descripcion": descripcion_list
    })

    # Si fila vacía -> ya tenemos (None, None) más abajo codigos_repetidos = None
    # --- (3) Patrón UT + 3 dígitos con posibles espacios ---
    pat_ut3 = re.compile(r"\bU\s*T\s*\d(?:\s*\d){2}\b", re.IGNORECASE)

    def _extract_UT3(txt: str):
        if pd.isna(txt) or txt is None:
            return None
        m = pat_ut3.search(str(txt))
        if not m:
            return None
        raw = m.group(0)
        # Normalizar: "U T 9 0 5" -> "UT905"
        return re.sub(r"\s+", "", raw.upper())

    # --- (4) Detectar códigos alfanuméricos repetidos ---
    pat_code = re.compile(
        r"\b(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{4,}\b"
    )

    def _find_codes(txt: str):
        if pd.isna(txt) or txt is None:
            return []
        return pat_code.findall(str(txt).upper())

    # Frecuencias globales de tokens en 'descripcion'
    all_tokens = []
    for t in out["descripcion"]:
        all_tokens.extend(_find_codes(t))
    freq = Counter(all_tokens)
    repeated_set = {tok for tok, cnt in freq.items() if cnt >= 2}

    def _codes_repeated_in_row(txt: str):
        toks = set(_find_codes(txt))
        hits = sorted(toks.intersection(repeated_set))
        return hits if hits else None

    out["codigos_repetidos"] = out["descripcion"].apply(_codes_repeated_in_row)

    # --- (5) Borrar de 'descripcion' el primer UT3 y todos los códigos repetidos ---
    def _remove_found(desc: str, repeated_codes):
        # Si la fila venía vacía, la dejamos igual
        if desc is None or pd.isna(desc):
            return desc
        s = str(desc)
        # Eliminar la PRIMERA ocurrencia del patrón UT3 flexible
        s = pat_ut3.sub(" ", s, count=1)
        # Eliminar cada código repetido detectado (como palabra completa, case-insensitive)
        if repeated_codes:
            for tok in repeated_codes:
                patt = re.compile(rf"\b{re.escape(tok)}\b", re.IGNORECASE)
                s = patt.sub(" ", s)
        # Limpieza final
        s = re.sub(r"\s+", " ", s).strip(" -_,.;: \t")
        return s

    out["descripcion"] = [
        _remove_found(d, rc) for d, rc in zip(out["descripcion"], out["codigos_repetidos"])
    ]

    # --- (6) Devolver solo las 3 columnas requeridas ---
    return out[["ubicacion", "descripcion", "codigos_repetidos"]]

# Maintenance Dashboard (Streamlit)

Interfaz para técnicos de mantenimiento con datos en CSV.

## Estructura
```
maintenance-dashboard/
├── app/                 # App Streamlit
│   ├── app.py
│   └── pages/           # Páginas multipágina
├── src/                 # Módulos reutilizables (ETL, utils, plots, etc.)
├── data/
│   ├── raw/             # CSV originales (solo lectura)
│   └── processed/       # Datos limpios / feature tables
├── notebooks/           # Exploración y prototipos
├── scripts/             # Scripts CLI (ETL, validación)
├── tests/               # Pruebas unitarias (pytest)
├── docs/                # Documentación adicional
├── .streamlit/          # Configuración de Streamlit
├── requirements.txt
├── .gitignore
└── README.md
```

## Arranque rápido
```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

## Variables de entorno
Crea un archivo `.env` en la raíz (opcional) o usa variables de entorno del sistema.
Consulta `.env.sample` para ver claves disponibles.

## Notebooks
Usa la carpeta `notebooks/`. Hay un notebook de ejemplo: `exploracion.ipynb`.

## Pruebas
```bash
pytest -q
```

## Licencia
MIT

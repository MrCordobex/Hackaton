#!/usr/bin/env bash
set -euo pipefail

# ========= CONFIG =========
MODEL_PATH="../models/modelos_baseline_clav_act.joblib"
PORT_GENERATOR=8501
PORT_FRONTEND=8502
DEPS="streamlit graphviz joblib"
# =========================

echo "==========================================================="
echo "🧠 Lanzando aplicaciones Streamlit"
echo "   incident_generator.py → http://localhost:${PORT_GENERATOR}"
echo "   frontend.py           → http://localhost:${PORT_FRONTEND}"
echo "==========================================================="
echo

# 1) Binario Graphviz del sistema
if ! command -v dot &>/dev/null; then
  echo "⚠️  Instalando graphviz (binario del sistema)..."
  sudo apt update && sudo apt install -y graphviz
else
  echo "✅ Graphviz (sistema): $(dot -V | head -n 1)"
fi

# 2) uv/uvx presentes
command -v uvx >/dev/null || { echo "❌ Falta uvx. Instala con: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
command -v uv >/dev/null  || { echo "❌ Falta uv.  Instala con: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

# 3) ¿Tenemos pyproject con dependencias?
use_project=false
if [[ -f "pyproject.toml" ]]; then
  if grep -q "streamlit" pyproject.toml && grep -q "graphviz" pyproject.toml; then
    use_project=true
    echo "📦 pyproject.toml detectado con dependencias -> usaré 'uv run'"
    # Asegura instalación local de deps (rápido por caché)
    uv sync -q || true
  fi
fi

# 4) Smoke test de graphviz (wrapper Python)
if $use_project; then
  uv run python - <<'PY' || { echo "❌ graphviz (Python) no disponible"; exit 1; }
import graphviz; print("✅ graphviz (Python):", graphviz.__version__)
PY
else
  uvx -q --with graphviz python - <<'PY' || { echo "❌ graphviz (Python) no disponible"; exit 1; }
import graphviz; print("✅ graphviz (Python):", graphviz.__version__)
PY
fi

echo
echo "🚀 Arrancando servicios…"

if $use_project; then
  # Modo proyecto: sin avisos
  MODEL_PATH="$MODEL_PATH" uv run streamlit run incident_generator.py --server.port $PORT_GENERATOR --server.headless true &
  PID_GENERATOR=$!
  uv run streamlit run frontend.py --server.port $PORT_FRONTEND --server.headless true &
  PID_FRONTEND=$!
else
  # Modo efímero: silenciado (-q) y deps con --with
  MODEL_PATH="$MODEL_PATH" uvx -q --with $DEPS streamlit run incident_generator.py \
      --server.port $PORT_GENERATOR --server.headless true &
  PID_GENERATOR=$!

  uvx -q --with $DEPS streamlit run frontend.py \
      --server.port $PORT_FRONTEND --server.headless true &
  PID_FRONTEND=$!
fi

echo
echo "✅ Corriendo:"
echo "   - Generador PID $PID_GENERATOR (:${PORT_GENERATOR})"
echo "   - Frontend  PID $PID_FRONTEND (:${PORT_FRONTEND})"
echo
echo "🛑 Ctrl+C para detener ambos."
trap "echo; echo '🧹 Deteniendo…'; kill $PID_GENERATOR $PID_FRONTEND 2>/dev/null || true" SIGINT
wait

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

run:
	streamlit run app/app.py

test:
	pytest -q

format:
	ruff check --select I --fix . || true
	ruff format . || true

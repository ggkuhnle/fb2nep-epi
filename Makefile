PY=python3
VENV=.venv

.PHONY: venv deps data validate clean

venv:
	$(PY) -m venv $(VENV)

deps:
	. $(VENV)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt scipy

data:
	. $(VENV)/bin/activate && python scripts/generate_dataset.py

validate:
	. $(VENV)/bin/activate && python scripts/validate_dataset.py

clean:
	rm -rf $(VENV) data/synthetic/fb2nep.csv


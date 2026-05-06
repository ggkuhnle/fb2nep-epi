PY=python3
VENV=.venv

.PHONY: venv deps data hippo-data validate clean rubric

venv:
	$(PY) -m venv $(VENV)

deps:
	. $(VENV)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt scipy

data:
	. $(VENV)/bin/activate && python scripts/generate_dataset.py

hippo-data:
	. $(VENV)/bin/activate && python scripts/generate_hippo_survey_data.py

validate:
	. $(VENV)/bin/activate && python scripts/validate_dataset.py

rubric:
	quarto render assessment/rubric_assessment_1.qmd --to pdf
	cp _site/assessment/rubric_assessment_1.pdf assessment/rubric_assessment_1.pdf

clean:
	rm -rf $(VENV) data/synthetic/fb2nep.csv


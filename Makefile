.PHONY: run test prep lint

run:
	streamlit run supplyradar/app/main.py

test:
	python -m pytest tests/ -q

prep:
	python -m supplyradar.prep.run --workbook $(WORKBOOK) --base-date $(BASE_DATE) --out ./out

lint:
	ruff check supplyradar tests

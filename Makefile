PY ?= python

.PHONY: install test bench lint selftest clean docker

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q -W ignore::UserWarning

bench:
	$(PY) -m activeforge.cli bench --out benchmark.json

lint:
	$(PY) -m compileall -q activeforge tests

selftest: lint test bench

clean:
	rm -rf __pycache__ .pytest_cache */__pycache__

docker:
	docker build -t activeforge:0.1.0 .
	docker run --rm -v $(PWD)/benchmark.json:/app/benchmark.json activeforge:0.1.0

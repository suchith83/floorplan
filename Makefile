# One command regenerates every reported number (work order 07): runs every capture (model outputs replay from
# out/_cache), fits the interval constants, reruns, and writes eval/BENCHMARK.md, eval/CALIBRATION.md,
# eval/benchmark.json and fp/calibration.json.
.PHONY: benchmark benchmark-tables test

benchmark:
	uv run python eval/run_benchmark.py --all

# Rebuild the tables from the plans already in out/ (no fp runs).
benchmark-tables:
	uv run python eval/run_benchmark.py

test:
	uv run pytest -q

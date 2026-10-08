# One command regenerates every reported number (work order 07): runs every capture (model outputs replay from
# out/_cache), fits the interval constants, reruns, and writes eval/BENCHMARK.md, eval/CALIBRATION.md,
# eval/benchmark.json and fp/calibration.json.
.PHONY: benchmark benchmark-tables test fix-before fix-after

benchmark:
	uv run python eval/run_benchmark.py --all

# Rebuild the tables from the plans already in out/ (no fp runs, no refit: fp/calibration.json is left as it is).
benchmark-tables:
	uv run python eval/run_benchmark.py --no-fit

test:
	uv run pytest -q

# Fix loop (work order 09): regenerate each side from raw inputs. Checks the tag out in a temporary worktree (data/
# and the model-output cache out/_cache linked in), runs the whole benchmark there, writes fixloop/<side>/.
fix-before:
	scripts/fixloop.sh fix-before fixloop/before

fix-after:
	scripts/fixloop.sh fix-after fixloop/after

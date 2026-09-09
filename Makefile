# genius-bar. `make eval` is the one target that must work with no API key.
.PHONY: help data taxonomy golden label recheck rate eval test reproduce clean-cache

help:  ## Show this list
	@grep -hE '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t14

data:  ## Rebuild data/apple_threads.parquet from the Kaggle dump (needs Kaggle creds)
	uv run python -m genius_bar.data

taxonomy:  ## Re-derive the intent taxonomy draft (needs embedding quota)
	uv run python scripts/derive_taxonomy.py --cached-only

golden:  ## Draw the 180-example golden set (unlabelled)
	uv run python scripts/sample_golden.py

label:  ## Hand-label the golden set: 60 blind, then 120 assisted
	uv run python -m genius_bar.label

recheck:  ## Re-label 30 examples to measure annotator self-consistency
	uv run python -m genius_bar.label --mode recheck --n 30

rate:  ## Hand-rate agent replies, to measure judge-vs-human agreement
	uv run python -m genius_bar.label --mode reply

eval:  ## Reproduce all headline results. Works with NO API key when cache is warm.
	uv run python -m genius_bar.eval

test:  ## Run the test suite
	uv run pytest -q

reproduce:  ## What a grader runs: install, test, eval. Target is under 15 minutes.
	uv sync
	uv run pytest -q
	uv run python -m genius_bar.eval

clean-cache:  ## Drop cached LLM responses. Deletes committed results -- rarely what you want.
	@echo "This deletes the committed cache that makes 'make eval' work offline."
	@read -p "Type 'yes' to continue: " a && [ "$$a" = yes ] && rm -rf cache/llm cache/embed

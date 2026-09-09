# genius-bar. `make eval` is the one target that must work with no API key.
.PHONY: help data test clean-cache

help:
	@grep -hE '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t20

data:  ## Rebuild data/apple_threads.parquet from the Kaggle dump (needs Kaggle creds)
	uv run python -m genius_bar.data

test:  ## Run the test suite
	uv run pytest -q

clean-cache:  ## Drop cached LLM responses. Deletes committed results -- rarely what you want.
	@echo "This deletes the committed cache that makes `make eval` work offline."
	@read -p "Type 'yes' to continue: " a && [ "$$a" = yes ] && rm -rf cache/llm cache/embed

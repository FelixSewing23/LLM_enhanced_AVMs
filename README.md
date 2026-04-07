# Can Language Models Improve Real Estate Valuation?


### Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for Jupyter
cp .env.example .env  # add API keys only if re-running data collection or LLM extraction
```

### Running the Analysis
The main notebook reproduces all results:
```bash
jupyter notebook notebooks/houston_zillow_llm_avm.ipynb
```

### Data
The data files are provided separately and must be placed in the `data/` directory before running the notebook. The expected structure is:

```
data/
  raw/
    detail_raw.json          # 455 MB, raw Zillow scrape via Apify
    zip_search_raw.json      # ZIP-level search results
  interim/
    listings_clean.parquet   # cleaned sample (1,652 listings)
    immediate_output.jsonl   # raw LLM API responses
    batch_input.jsonl        # OpenAI batch request file
  processed/
    listings_with_llm.parquet  # final dataset with LLM features (required for modeling)
    listings_clean.parquet     # cleaned baseline data
```

The notebook can be run starting from the modeling cells if only `data/processed/listings_with_llm.parquet` is available. Running the full pipeline from data collection requires API keys (Apify, OpenAI) configured in `.env`.


### License

This repository is provided for academic review purposes. All rights reserved.

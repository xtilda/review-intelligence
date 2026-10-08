# Review Intelligence

An English customer-feedback dashboard backed by a trained, reproducible sentiment baseline. Paste reviews or upload a CSV, inspect sentiment predictions, examine recurring product aspects, and export the results.

## Engineering

- TF-IDF unigrams/bigrams + logistic regression trained on open UCI data.
- Text deduplication before a stratified 80/20 split; conflicting labels removed.
- Versioned JSON model, held-out metrics, confusion matrix, domain breakdown, and individual errors.
- Dependency-light pure-Python inference verified against scikit-learn predictions (maximum probability difference: 2.22e-16).
- Uncertain predictions for out-of-vocabulary text or probability below 0.60. This is abstention, not a neutral class.
- Keyword aspect rules for delivery, quality, support, value, usability. These are **not** a learned topic model.
- Bounded CSV input, server-side validation, text-safe UI rendering, formula-safe CSV export, tests and GitHub Actions.
- Stateless FastAPI deployment: no API key, Neon, Docker, or persistent upload storage.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000. The trained model is included; no training required to deploy.

## Train and evaluate

```bash
pip install -r requirements-dev.txt
python training/train.py
python -m pytest -q
```

Training downloads the source archive. An offline copy can be supplied with `--archive PATH`. The script rebuilds `models/sentiment.json`, `evaluation/report.json`, and `evaluation/errors.json`, then checks serving parity on all held-out sentences plus OOV/repeated-word probes. Fixed seed: 42. Training libraries are pinned for reproduction; runtime avoids numpy/scipy/sklearn.

Held-out results: **80.03% accuracy, 0.8003 macro F1**, 2,383 training / 596 test sentences. Results apply to this split only; confidence is uncalibrated, the dataset is balanced short English sentences, and the mixed-domain random split does not measure unseen-domain generalization. The UI abstention threshold was chosen heuristically; these test metrics describe the binary classifier before abstention.

## Data attribution

Kotzias, D. (2015). *Sentiment Labelled Sentences*. UCI Machine Learning Repository. https://doi.org/10.24432/C57604. Dataset licensed CC BY 4.0. Source: https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences. The generated model and evaluation error excerpts derive from this dataset; preprocessing combines two wrapped lines, normalizes text for deduplication, and removes conflicts. No raw archive is bundled.

## Deployment

See [VERCEL_KURULUM.md](VERCEL_KURULUM.md). App source and this project's model artifact are MIT licensed; dataset-derived excerpts retain their source attribution/license.

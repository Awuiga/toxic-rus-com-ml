# Toxic Russian Comments

[Русский](README.md) · **English**

A PyTorch exercise in preparing Russian comments for toxicity classification. The current pipeline cleans and splits the dataset, builds a vocabulary and data loaders, then passes one batch through an embedding layer. Classifier training, quality metrics, and toxicity predictions remain future work.

## Data pipeline

`prepare_data.py` reads a CSV with `comment` and `toxic` columns. Labels are 0 or 1. It drops empty rows and exact duplicates, lowercases the text, normalizes whitespace, and tokenizes words with a regular expression.

Each class is split into approximately 80% train, 10% validation, and 10% test with seed 42. The train-only vocabulary has up to 20,000 tokens and requires a minimum frequency of 2. `<PAD>` has ID 0 and `<UNK>` has ID 1. Each sequence is truncated or padded to 64 tokens. Batch size is 32. The script saves the vocabulary locally to `data/vocab.json`.

## Run on Windows

Verified with Python 3.13.2.

```powershell
git clone https://github.com/Awuiga/toxic-rus-com-ml.git
cd toxic-rus-com-ml
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Supply an authorized CSV as `data/labeled.csv`. [Expected format](data/README.md)

```powershell
python prepare_data.py
python train.py
```

The first script prints split, vocabulary, and batch sizes. The second runs the embedding step. Calling `build_loaders()` rebuilds the vocabulary.

## Limitations

Deduplication keeps the first label for repeated text. Conflicting labels and near-duplicates receive no separate handling, and class balance after cleanup is not evaluated. The split seed is fixed while embedding initialization is random. The original local CSV could not be tied to a confirmed source or license. Supply a compatible CSV from a source you may use. The corpus and derived vocabulary are local artifacts. [Verification](VERIFICATION.md)

# Information Retrieval Search Engine

Original Python source recovered from the uploaded CSCE 5200 Phase 3 submission.

## Features

- TREC-style document parsing, stopword filtering, and Porter stemming.
- Forward and inverted indexes with term frequencies.
- TF-IDF vector-space retrieval and cosine-similarity ranking.
- Title, description, and narrative query configurations.
- Precision and recall evaluation with user-supplied relevance judgments.

## Run

Use Python 3.10 or later. Run commands from this directory:

```bash
python -m pip install -r requirements.txt
python run_project.py ./ft911
```

Place your authorized document collection in `ft911/`. The indexer opens an interactive term lookup after building indexes; type `exit` to close it. Supply your own `topics.txt` and `main.qrels` for batch retrieval:

```bash
python query_processor.py --all
```

Run `python query_processor.py --help` for alternative input paths. The document collection, derived full-corpus indexes, and relevance files are not redistributed here. Synthetic smoke tests can verify the code without the original corpus.

## Provenance

Source recovered from `MominKhizarUddin_Mohammed_IR_Phase3.zip` in the user-provided submission export. This is the original uploaded Python implementation, separate from the reconstructed Java demo elsewhere in this repository. No original dataset evaluation has been rerun during portfolio import.

# Java Information Retrieval Search Engine — reconstructed

Parse UTF-8 plain text and basic TREC SGML with `<DOC>` / `<DOCNO>` blocks, build forward/inverted indices, and rank documents using TF-IDF cosine similarity. Document IDs preserve DOCNO where present.

Requires JDK 11 or newer. Source-file mode avoids a separate compile command:

```sh
java SearchEngine.java index sample_docs artifacts
java SearchEngine.java search artifacts "graph search robot"
```

Outputs: `vocabulary.txt`, `docIDs.txt`, `postings.txt`, `forward_index.txt`, `inverted_index.txt`, and serialized `index.bin`. The plain-text outputs are tab-separated; index.bin is for the search command. Duplicate document IDs fail explicitly.

This reconstruction uses simple suffix normalization, **not Porter stemming**. It does not implement qrels relevance evaluation, weighted-zone retrieval, or the original corpus parser's complete SGML handling. Only load serialized indices created by this application from trusted files.

from collections import defaultdict
from nltk.stem import PorterStemmer


class Indexer:
    """Builds and queries forward and inverted indexes."""

    def __init__(self, documents: dict, term_dict: dict, doc_dict: dict,
                 stopwords: set = None):

        self.documents = documents
        self.term_dict = term_dict
        self.doc_dict  = doc_dict
        self.stopwords = stopwords or set()
        self.stemmer   = PorterStemmer()

        self.forward_index  = {}
        self.inverted_index = defaultdict(dict)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_index(self) -> None:
        print("Building forward and inverted indexes...")

        for doc_id_str, tokens in self.documents.items():
            doc_numeric = self.doc_dict[doc_id_str]

            freq = defaultdict(int)
            for token in tokens:
                word_id = self.term_dict.get(token)
                if word_id is None:
                    continue
                freq[word_id] += 1

            self.forward_index[doc_numeric] = dict(freq)

            for word_id, count in freq.items():
                self.inverted_index[word_id][doc_numeric] = count

        print(f"  Forward index:  {len(self.forward_index)} documents")
        print(f"  Inverted index: {len(self.inverted_index)} unique terms")

    # ------------------------------------------------------------------
    # Write to disk
    # ------------------------------------------------------------------

    def write_forward_index(self, path: str = "forward_index.txt") -> None:
        with open(path, "w", encoding="utf-8") as f:
            for doc_id in sorted(self.forward_index):
                postings = self.forward_index[doc_id]
                parts = "; ".join(
                    f"{wid}: {cnt}"
                    for wid, cnt in sorted(postings.items())
                )
                f.write(f"{doc_id}: {parts}\n")
        print(f"  [OK] {path}")

    def write_inverted_index(self, path: str = "inverted_index.txt") -> None:
        with open(path, "w", encoding="utf-8") as f:
            for word_id in sorted(self.inverted_index):
                postings = self.inverted_index[word_id]
                parts = "; ".join(
                    f"{did}: {cnt}"
                    for did, cnt in sorted(postings.items())
                )
                f.write(f"{word_id}: {parts}\n")
        print(f"  [OK] {path}")

    def write_all(self,
                  fwd_path: str = "forward_index.txt",
                  inv_path: str = "inverted_index.txt") -> None:
        self.write_forward_index(fwd_path)
        self.write_inverted_index(inv_path)

    # ------------------------------------------------------------------
    # Query 
    # ------------------------------------------------------------------

    def query(self, raw_term: str) -> None:
        raw_term = raw_term.strip().lower()

        if not raw_term:
            return

        # Stopword check
        if raw_term in self.stopwords:
            print("This is a stopword. Please enter a meaningful term.\n")
            return

        # Stem
        stem = self.stemmer.stem(raw_term)

        # Get word ID
        word_id = self.term_dict.get(stem)
        if word_id is None:
            print(f"{raw_term} not found.\n")
            return

        postings = self.inverted_index.get(word_id, {})
        if not postings:
            print(f"{raw_term} not found.\n")
            return

        # Format output EXACTLY as required
        postings_str = "; ".join(
            f"{doc_id}:{freq}"
            for doc_id, freq in sorted(postings.items())
        )

        print(f"{raw_term} ({word_id}): {postings_str}\n")
"""
CSCE 5200 Project Phase 3 - Query Retrieval and Performance Analysis
Author: Mohammed Momin Khizar Uddin

Implements Vector Space Model retrieval using TF-IDF weighting and cosine similarity.
Reads Phase 2 parser_output.txt and inverted_index.txt, plus Project 3 topics.txt/main.qrels.

Output required by assignment:
    vsm_output.txt with TAB-separated fields:
    TOPIC    DOCUMENT    UNIQUE#    COSINE_VALUE

Run:
    python query_processor.py --setting title
    python query_processor.py --all
"""

import argparse
import math
import os
import re
from collections import defaultdict, Counter

try:
    from nltk.stem import PorterStemmer
    _stemmer = PorterStemmer()
    def stem_token(token: str) -> str:
        return _stemmer.stem(token)
except Exception:
    # Small fallback so the program still runs if NLTK is unavailable.
    def stem_token(token: str) -> str:
        for suffix in ("ingly", "edly", "ing", "ed", "ies", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                if suffix == "ies":
                    return token[:-3] + "i"
                return token[:-len(suffix)]
        return token


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PARSER_OUTPUT = os.path.join(BASE_DIR, "parser_output.txt")
DEFAULT_INVERTED_INDEX = os.path.join(BASE_DIR, "inverted_index.txt")
DEFAULT_STOPWORDS = os.path.join(BASE_DIR, "stopwordlist.txt")
DEFAULT_TOPICS = os.path.join(BASE_DIR, "topics.txt")
DEFAULT_QRELS = os.path.join(BASE_DIR, "main.qrels")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "vsm_output.txt")


def load_stopwords(path: str) -> set[str]:
    stopwords = set()
    if not os.path.exists(path):
        return stopwords
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                stopwords.add(word)
    return stopwords


def load_parser_output(path: str):
    """Load word dictionary and document dictionary from Phase 2 parser_output.txt."""
    term_dict: dict[str, int] = {}
    doc_dict: dict[str, int] = {}
    reverse_doc_dict: dict[int, str] = {}

    section = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("WORD DICTIONARY"):
                section = "word"
                continue
            if upper.startswith("DOCUMENT DICTIONARY"):
                section = "doc"
                continue

            parts = line.split()
            if len(parts) < 2:
                continue
            key, val = parts[0], parts[1]
            if not val.isdigit():
                continue

            if section == "word":
                term_dict[key] = int(val)
            elif section == "doc":
                doc_dict[key] = int(val)
                reverse_doc_dict[int(val)] = key

    return term_dict, doc_dict, reverse_doc_dict


def load_inverted_index(path: str):
    """Load inverted index formatted as termID: docID: freq; docID: freq; ..."""
    inverted_index: dict[int, dict[int, int]] = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            term_part, postings_part = line.split(":", 1)
            term_id = int(term_part.strip())
            postings: dict[int, int] = {}

            for entry in postings_part.split(";"):
                entry = entry.strip()
                if not entry or ":" not in entry:
                    continue
                doc_str, freq_str = entry.split(":", 1)
                doc_id = int(doc_str.strip())
                freq = int(freq_str.strip())
                postings[doc_id] = freq

            inverted_index[term_id] = postings

    return inverted_index


def load_topics(path: str):
    """Load TREC-style topics using <top>, <num>, <title>, <desc>, and <narr>."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    topics = []
    blocks = re.findall(r"<top>(.*?)</top>", content, flags=re.DOTALL | re.IGNORECASE)
    for block in blocks:
        num_match = re.search(r"<num>\s*Number:\s*(\d+)", block, flags=re.IGNORECASE)
        title_match = re.search(r"<title>\s*(.*?)(?=<desc>|<narr>|$)", block, flags=re.DOTALL | re.IGNORECASE)
        desc_match = re.search(r"<desc>\s*Description:\s*(.*?)(?=<narr>|$)", block, flags=re.DOTALL | re.IGNORECASE)
        narr_match = re.search(r"<narr>\s*Narrative:\s*(.*?)$", block, flags=re.DOTALL | re.IGNORECASE)

        topics.append({
            "num": int(num_match.group(1)) if num_match else 0,
            "title": title_match.group(1).strip() if title_match else "",
            "desc": desc_match.group(1).strip() if desc_match else "",
            "narr": narr_match.group(1).strip() if narr_match else "",
        })
    return topics


def load_qrels(path: str):
    qrels: dict[int, dict[str, int]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            parts = raw_line.split()
            if len(parts) < 4:
                continue
            topic = int(parts[0])
            doc_name = parts[2]
            relevance = int(parts[3])
            qrels[topic][doc_name] = relevance
    return qrels


def tokenize(text: str):
    # Course rules: lowercase, split on non-alphanumeric, ignore tokens containing digits.
    tokens = re.split(r"[^A-Za-z0-9]+", text.lower())
    return [t for t in tokens if t and not any(ch.isdigit() for ch in t)]


def preprocess_text(text: str, stopwords: set[str], term_dict: dict[str, int]):
    term_freq = Counter()
    for token in tokenize(text):
        if token in stopwords:
            continue
        stemmed = stem_token(token)
        term_id = term_dict.get(stemmed)
        if term_id is not None:
            term_freq[term_id] += 1
    return dict(term_freq)


def idf(term_id: int, inverted_index: dict[int, dict[int, int]], total_docs: int):
    df = len(inverted_index.get(term_id, {}))
    if df == 0:
        return 0.0
    return math.log10(total_docs / df)


def tf_weight(tf: int):
    if tf <= 0:
        return 0.0
    return 1.0 + math.log10(tf)


def build_document_lengths(inverted_index: dict[int, dict[int, int]], total_docs: int):
    lengths_sq = defaultdict(float)
    for term_id, postings in inverted_index.items():
        term_idf = idf(term_id, inverted_index, total_docs)
        for doc_id, tf in postings.items():
            weight = tf_weight(tf) * term_idf
            lengths_sq[doc_id] += weight * weight
    return {doc_id: math.sqrt(value) for doc_id, value in lengths_sq.items()}


def retrieve(query_tf: dict[int, int], inverted_index, doc_lengths, total_docs: int, top_n: int):
    if not query_tf:
        return []

    query_weights = {}
    query_length_sq = 0.0
    for term_id, qtf in query_tf.items():
        weight = tf_weight(qtf) * idf(term_id, inverted_index, total_docs)
        query_weights[term_id] = weight
        query_length_sq += weight * weight

    query_length = math.sqrt(query_length_sq)
    if query_length == 0:
        return []

    scores = defaultdict(float)
    for term_id, query_weight in query_weights.items():
        term_idf = idf(term_id, inverted_index, total_docs)
        for doc_id, tf in inverted_index.get(term_id, {}).items():
            doc_weight = tf_weight(tf) * term_idf
            scores[doc_id] += query_weight * doc_weight

    results = []
    for doc_id, dot_product in scores.items():
        doc_length = doc_lengths.get(doc_id, 0.0)
        if doc_length > 0:
            cosine = dot_product / (query_length * doc_length)
            results.append((doc_id, cosine))

    results.sort(key=lambda item: (-item[1], item[0]))
    return results[:top_n]


def query_text(topic: dict, setting: str):
    if setting == "title":
        return topic["title"]
    if setting == "desc":
        return topic["title"] + " " + topic["desc"]
    if setting == "narr":
        return topic["title"] + " " + topic["narr"]
    raise ValueError("setting must be title, desc, or narr")


def evaluate(results, topic_num: int, qrels, reverse_doc_dict):
    relevant_docs = {doc for doc, rel in qrels.get(topic_num, {}).items() if rel > 0}
    total_relevant = len(relevant_docs)
    retrieved_docs = [reverse_doc_dict.get(doc_id, f"UNKNOWN-{doc_id}") for doc_id, _ in results]
    retrieved_relevant = sum(1 for doc in retrieved_docs if doc in relevant_docs)

    precision = retrieved_relevant / len(retrieved_docs) if retrieved_docs else 0.0
    recall = retrieved_relevant / total_relevant if total_relevant else 0.0

    def precision_at(k):
        subset = retrieved_docs[:k]
        if not subset:
            return 0.0
        return sum(1 for doc in subset if doc in relevant_docs) / len(subset)

    def recall_at(k):
        if total_relevant == 0:
            return 0.0
        subset = retrieved_docs[:k]
        return sum(1 for doc in subset if doc in relevant_docs) / total_relevant

    return {
        "total_relevant": total_relevant,
        "retrieved": len(retrieved_docs),
        "retrieved_relevant": retrieved_relevant,
        "precision": precision,
        "recall": recall,
        "p10": precision_at(10),
        "r10": recall_at(10),
        "p100": precision_at(100),
        "r100": recall_at(100),
    }


def write_output(results_by_topic, reverse_doc_dict, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for topic_num in sorted(results_by_topic):
            for rank, (doc_id, score) in enumerate(results_by_topic[topic_num], start=1):
                doc_name = reverse_doc_dict.get(doc_id, f"UNKNOWN-{doc_id}")
                f.write(f"{topic_num}\t{doc_name}\t{rank}\t{score:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="CSCE 5200 Project 3 Query Processor")
    parser.add_argument("--parser-output", default=DEFAULT_PARSER_OUTPUT)
    parser.add_argument("--inverted-index", default=DEFAULT_INVERTED_INDEX)
    parser.add_argument("--stopwords", default=DEFAULT_STOPWORDS)
    parser.add_argument("--topics", default=DEFAULT_TOPICS)
    parser.add_argument("--qrels", default=DEFAULT_QRELS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--setting", choices=["title", "desc", "narr"], default="title")
    parser.add_argument("--topn", type=int, default=1000)
    parser.add_argument("--all", action="store_true", help="Run title, desc, and narr settings and create separate output files")
    args = parser.parse_args()

    stopwords = load_stopwords(args.stopwords)
    term_dict, doc_dict, reverse_doc_dict = load_parser_output(args.parser_output)
    inverted_index = load_inverted_index(args.inverted_index)
    topics = load_topics(args.topics)
    qrels = load_qrels(args.qrels)
    total_docs = len(doc_dict)

    print(f"Loaded {len(term_dict)} terms and {total_docs} documents.")
    print(f"Loaded {len(inverted_index)} inverted-index entries and {len(topics)} topics.")
    print("Precomputing document lengths...")
    doc_lengths = build_document_lengths(inverted_index, total_docs)

    settings = ["title", "desc", "narr"] if args.all else [args.setting]
    summary_rows = []

    for setting in settings:
        results_by_topic = {}
        print(f"\nSETTING: {setting}")
        for topic in topics:
            topic_num = topic["num"]
            processed_query = preprocess_text(query_text(topic, setting), stopwords, term_dict)
            ranked_results = retrieve(processed_query, inverted_index, doc_lengths, total_docs, args.topn)
            results_by_topic[topic_num] = ranked_results
            metrics = evaluate(ranked_results, topic_num, qrels, reverse_doc_dict)
            summary_rows.append((setting, topic_num, metrics))
            print(
                f"Topic {topic_num}: retrieved={metrics['retrieved']}, "
                f"relevant_found={metrics['retrieved_relevant']}/{metrics['total_relevant']}, "
                f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
                f"P@10={metrics['p10']:.4f}, R@100={metrics['r100']:.4f}"
            )

        if args.all:
            out_name = os.path.splitext(args.output)[0] + f"_{setting}.txt"
        else:
            out_name = args.output
        write_output(results_by_topic, reverse_doc_dict, out_name)
        print(f"Wrote {out_name}")

    summary_path = os.path.join(os.path.dirname(args.output), "evaluation_summary.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("setting,topic,total_relevant,retrieved,retrieved_relevant,precision,recall,p10,r10,p100,r100\n")
        for setting, topic_num, m in summary_rows:
            f.write(
                f"{setting},{topic_num},{m['total_relevant']},{m['retrieved']},{m['retrieved_relevant']},"
                f"{m['precision']:.6f},{m['recall']:.6f},{m['p10']:.6f},{m['r10']:.6f},{m['p100']:.6f},{m['r100']:.6f}\n"
            )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

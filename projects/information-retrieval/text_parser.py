import os
import re
import sys
from nltk.stem import PorterStemmer


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

def load_stopwords(path: str) -> set:
    stopwords = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    stopwords.add(word)
    except FileNotFoundError:
        print(f"[WARNING] Stopword file not found: {path}. Proceeding with empty list.")
    return stopwords


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list:
    text = text.lower()
    tokens = re.split(r"[^a-zA-Z0-9]+", text)

    result = []
    for token in tokens:
        if not token:
            continue
        if token.isdigit():
            continue
        if any(c.isdigit() for c in token):
            continue
        result.append(token)

    return result


# ---------------------------------------------------------------------------
# TREC Document Parser
# ---------------------------------------------------------------------------

def parse_dataset(folder: str, stopwords: set, stemmer: PorterStemmer) -> dict:
    documents = {}

    files = sorted(
        os.listdir(folder),
        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else x
    )

    for fname in files:
        path = os.path.join(folder, fname)

        if not os.path.isfile(path):
            continue

        with open(path, "r", encoding="latin1") as f:
            content = f.read()

        doc_blocks = re.findall(r"<DOC>(.*?)</DOC>", content, re.DOTALL)

        for doc_block in doc_blocks:
            docno_match = re.search(r"<DOCNO>(.*?)</DOCNO>", doc_block, re.DOTALL)
            if not docno_match:
                continue

            doc_id = docno_match.group(1).strip()

            text_blocks = re.findall(r"<TEXT>(.*?)</TEXT>", doc_block, re.DOTALL)
            text = " ".join(text_blocks)

            tokens = tokenize(text)

            processed = [
                stemmer.stem(token)
                for token in tokens
                if token not in stopwords
            ]

            if processed:
                documents[doc_id] = processed

    return documents


# ---------------------------------------------------------------------------
# Test Data Parser
# Supports lines like:
# FT911-3328 some text...
# T911-3328 some text...
# ---------------------------------------------------------------------------

def parse_test_file(file_path: str, stopwords: set, stemmer: PorterStemmer) -> dict:
    documents = {}

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    doc_counter = 1

    for line in lines:
        line = line.strip()

        if not line:
            continue

        match = re.match(r"([A-Z]*\d+-\d+)\s+(.*)", line)

        if match:
            doc_id = match.group(1).strip()
            text = match.group(2).strip()
        else:
            doc_id = f"DOC-{doc_counter}"
            text = line
            doc_counter += 1

        tokens = tokenize(text)

        processed = [
            stemmer.stem(token)
            for token in tokens
            if token not in stopwords
        ]

        if processed:
            documents[doc_id] = processed

    return documents


# ---------------------------------------------------------------------------
# Dictionary builders
# ---------------------------------------------------------------------------

def build_term_dict(documents: dict) -> dict:
    all_terms = set()

    for tokens in documents.values():
        all_terms.update(tokens)

    return {term: i for i, term in enumerate(sorted(all_terms), 1)}


def build_doc_dict(documents: dict) -> dict:
    doc_dict = {}

    sorted_docs = sorted(
        documents.keys(),
        key=lambda x: int(re.search(r'(\d+)$', x).group(1)) if re.search(r'(\d+)$', x) else 0
    )

    for i, doc_id in enumerate(sorted_docs, start=1):
        doc_dict[doc_id] = i

    return doc_dict


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_parser_output(term_dict: dict, doc_dict: dict,
                        path: str = "parser_output.txt") -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("WORD DICTIONARY\n\n")

        for term, tid in sorted(term_dict.items(), key=lambda x: x[1]):
            f.write(f"{term}\t{tid}\n")

        f.write("\nDOCUMENT DICTIONARY\n\n")

        for doc_id, did in sorted(doc_dict.items(), key=lambda x: x[1]):
            f.write(f"{doc_id}\t{did}\n")

    print(f"  [OK] {path}")


def write_documents(documents: dict, doc_dict: dict,
                    path: str = "documents.txt") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for doc_id in sorted(doc_dict, key=doc_dict.get):
            f.write(doc_id + "\t" + " ".join(documents[doc_id]) + "\n")

    print(f"  [OK] {path}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_parser(data_folder: str = "./ft911",
               stopwords_path: str = "./stopwordlist.txt") -> tuple:

    if data_folder == "./ft911" and len(sys.argv) > 1:
        data_folder = sys.argv[1]

    print("Loading stopwords...")
    stopwords = load_stopwords(stopwords_path)
    print(f"  {len(stopwords)} stopwords loaded.")

    stemmer = PorterStemmer()

    is_test = os.path.isfile(data_folder)

    if is_test:
        print(f"Parsing TEST FILE: '{data_folder}'...")
        documents = parse_test_file(data_folder, stopwords, stemmer)
    else:
        print(f"Parsing documents from '{data_folder}'...")
        documents = parse_dataset(data_folder, stopwords, stemmer)

    print(f"  {len(documents)} documents parsed.")

    print("Building dictionaries...")
    term_dict = build_term_dict(documents)
    doc_dict = build_doc_dict(documents)

    print(f"  {len(term_dict)} unique terms.")

    print("Writing parser output files...")
    if is_test:
        write_parser_output(term_dict, doc_dict, "parser_output_test.txt")
        write_documents(documents, doc_dict, "documents_test.txt")
    else:
        write_parser_output(term_dict, doc_dict, "parser_output.txt")
        write_documents(documents, doc_dict, "documents.txt")

    return documents, term_dict, doc_dict, stopwords


# ---------------------------------------------------------------------------
# Standalone run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_parser()
    print("\nDone!")
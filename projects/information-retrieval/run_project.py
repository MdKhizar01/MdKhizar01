# run_project.py

import os
import sys
import time
from text_parser import run_parser
from indexer import Indexer


def main():
    print("====================================")
    print("      PROJECT PHASE 2 START         ")
    print("====================================\n")

    # Detect input path
    data_path = sys.argv[1] if len(sys.argv) > 1 else "./ft911"
    is_test = os.path.isfile(data_path)

    # Start timing
    start_time = time.time()

    # Step 1: Parse documents
    documents, term_dict, doc_dict, stopwords = run_parser()

    # Step 2: Build indexes
    indexer = Indexer(documents, term_dict, doc_dict, stopwords)
    indexer.build_index()

    # Step 3: Write index files
    if is_test:
        indexer.write_all("forward_index_test.txt", "inverted_index_test.txt")
    else:
        indexer.write_all("forward_index.txt", "inverted_index.txt")

    # End timing
    end_time = time.time()
    indexing_time = end_time - start_time
    print(f"\nIndexing Time: {indexing_time:.2f} seconds")

    # Step 4: Interactive query mode
    print("\n====================================")
    print("        QUERY MODE STARTED          ")
    print("====================================")
    print("Type a word to search.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            query = input("Enter query term: ").strip().lower()

            if query == "exit":
                print("Exiting... Goodbye!")
                break

            if not query:
                continue

            if query in stopwords:
                print("This is a stopword. Please enter a meaningful term.\n")
                continue

            indexer.query(query)

        except KeyboardInterrupt:
            print("\nExiting... Goodbye!")
            break


if __name__ == "__main__":
    main()
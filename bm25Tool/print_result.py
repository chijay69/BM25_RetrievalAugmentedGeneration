"""print_result.py"""

import os
from itertools import groupby
from typing import List, Tuple

from bm25Tool.build_document_index import read_file_content
from converter.Document import Document


def print_results(results: List[Tuple[Document, float]], output_directory: str, show_full_text: bool):
    """Prints the search results."""
    results.sort(key=lambda doc_: doc_[0].metadata['filename'])
    grouped_result = groupby(results, key=lambda doc_: doc_[0].metadata['filename'])

    for doc_name, group in grouped_result:
        print(f"========================= {doc_name} ============================")
        toc_filename: str = doc_name.rsplit(".", 1)[0] + "_toc.md"
        toc_file_path = os.path.join(output_directory, toc_filename)
        if os.path.exists(toc_file_path):
            toc_content: str = read_file_content(toc_file_path)
            print("Table of Content\n" + toc_content + "\n=====\n")

        for doc, score in group:
            snippet_content = "\n".join(doc.chunk_content.split("\n")[3:])
            snippet = snippet_content if show_full_text else snippet_content[:100] + '...' + snippet_content[-100:]
            print(f"Relevant Snippet:\n{snippet}\nScore: {score:.1f}\n=====\n")

        print("\n============\n")
